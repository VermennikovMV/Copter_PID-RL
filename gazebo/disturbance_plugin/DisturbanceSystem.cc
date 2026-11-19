// World system: scheduled external force on one vehicle link and a scheduled payload release.
//
// A schedule arrives as gz.msgs.StringMsg on <topic>, one item per line:
//   force <on_s> <off_s> <fx_N> <fy_N> <fz_N>     world-frame (ENU) force at the link centre of mass
//   detach <t_s>                                  publish gz.msgs.Empty on <detach_topic> once
// Times are simulation seconds counted from the first physics update after the schedule
// was received, so the profile follows Gazebo time exactly, independent of wall-clock delays.
// Every change of the applied force and the release are written to <log_file> as JSON lines.
//
// Optional wind visualisation (<wind_streaks> on one instance): the models <prefix>0..N-1 (thin
// visual-only streaks) drift with the wind velocity = applied horizontal force / <drag_N_per_mps>
// inside the box <area_min>..<area_max>; without wind they are parked below the ground.
#include <gz/msgs/empty.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/transport/Node.hh>
#include <sdf/Element.hh>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace copter_disturbance {

struct ForceEvent {
  double on, off;
  gz::math::Vector3d force;
};

struct Schedule {
  std::vector<ForceEvent> forces;
  std::optional<double> detach;
};

class DisturbanceSystem : public gz::sim::System,
                          public gz::sim::ISystemConfigure,
                          public gz::sim::ISystemPreUpdate {
  gz::transport::Node node;
  gz::transport::Node::Publisher detachPub;
  std::mutex mutex;
  std::optional<Schedule> pending;
  Schedule active;
  bool armed = false, detached = false;
  double t0 = 0.;
  std::string modelName, linkName;
  gz::sim::Entity link = gz::sim::kNullEntity;
  gz::math::Vector3d lastForce{0, 0, 0};
  std::ofstream log;
  // wind visualisation
  std::string streakPrefix;
  std::vector<gz::sim::Entity> streakModels;
  std::vector<gz::math::Vector3d> streakPos;
  gz::math::Vector3d areaMin, areaMax;
  double dragCoefficient = 1.52, lastStreakUpdate = -1.;
  bool streaksParked = false;

 public:
  void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &sdf,
                 gz::sim::EntityComponentManager &, gz::sim::EventManager &) override {
    modelName = sdf->Get<std::string>("model_name", "drone1").first;
    linkName = sdf->Get<std::string>("link_name", "base_link").first;
    const auto topic = sdf->Get<std::string>("topic", "/copter/disturbance").first;
    const auto detachTopic = sdf->Get<std::string>("detach_topic", "/copter/payload/detach").first;
    if (sdf->HasElement("log_file")) {
      log.open(sdf->Get<std::string>("log_file"));
      log << std::setprecision(10);
    }
    detachPub = node.Advertise<gz::msgs::Empty>(detachTopic);
    if (sdf->HasElement("wind_streaks")) {
      auto w = sdf->FindElement("wind_streaks");
      streakPrefix = w->Get<std::string>("prefix", "wind_streak_").first;
      const int count = w->Get<int>("count", 0).first;
      areaMin = w->Get<gz::math::Vector3d>("area_min", gz::math::Vector3d(-10, -10, 1)).first;
      areaMax = w->Get<gz::math::Vector3d>("area_max", gz::math::Vector3d(10, 10, 5)).first;
      dragCoefficient = w->Get<double>("drag_N_per_mps", 1.52).first;
      std::mt19937 rng(7);
      std::uniform_real_distribution<double> u(0., 1.);
      for (int k = 0; k < count; ++k) {
        streakPos.emplace_back(areaMin.X() + u(rng) * (areaMax.X() - areaMin.X()),
                               areaMin.Y() + u(rng) * (areaMax.Y() - areaMin.Y()),
                               areaMin.Z() + u(rng) * (areaMax.Z() - areaMin.Z()));
        streakModels.push_back(gz::sim::kNullEntity);
      }
    }
    node.Subscribe(topic, &DisturbanceSystem::Receive, this);
    Write("{\"event\":\"configured\",\"topic\":\"" + topic + "\",\"detach_topic\":\"" + detachTopic + "\"}");
  }

  void Receive(const gz::msgs::StringMsg &message) {
    Schedule schedule;
    std::istringstream lines(message.data());
    std::string line;
    while (std::getline(lines, line)) {
      std::istringstream item(line);
      std::string kind;
      if (!(item >> kind)) continue;
      if (kind == "force") {
        ForceEvent e;
        double x, y, z;
        if (!(item >> e.on >> e.off >> x >> y >> z) || !std::isfinite(x + y + z + e.on + e.off)) return;
        e.force.Set(x, y, z);
        schedule.forces.push_back(e);
      } else if (kind == "detach") {
        double t;
        if (!(item >> t) || !std::isfinite(t)) return;
        schedule.detach = t;
      } else {
        return;   // unknown item: reject the whole schedule
      }
    }
    std::lock_guard<std::mutex> guard(mutex);
    pending = schedule;
  }

  void PreUpdate(const gz::sim::UpdateInfo &info, gz::sim::EntityComponentManager &ecm) override {
    if (info.paused) return;
    const double now = std::chrono::duration<double>(info.simTime).count();
    {
      std::lock_guard<std::mutex> guard(mutex);
      if (pending) {
        active = *pending;
        pending.reset();
        armed = true;
        detached = false;
        t0 = now;
        std::ostringstream s;
        s << "{\"event\":\"schedule_armed\",\"sim_time\":" << now << ",\"forces\":" << active.forces.size()
          << ",\"detach\":" << (active.detach ? *active.detach : -1.) << "}";
        Write(s.str());
      }
    }
    if (!armed) return;
    if (link == gz::sim::kNullEntity) {
      auto model = ecm.EntityByComponents(gz::sim::components::Name(modelName), gz::sim::components::Model());
      if (model == gz::sim::kNullEntity) return;
      // the vehicle model includes the iris model; search all descendant links by name
      for (auto candidate : ecm.Descendants(model)) {
        auto name = ecm.Component<gz::sim::components::Name>(candidate);
        if (name && name->Data() == linkName && gz::sim::Link(candidate).Valid(ecm)) {
          link = candidate;
          break;
        }
      }
      if (link == gz::sim::kNullEntity) return;
      if (!ecm.Component<gz::sim::components::WorldPose>(link))
        ecm.CreateComponent(link, gz::sim::components::WorldPose());
    }
    const double t = now - t0;
    gz::math::Vector3d force{0, 0, 0};
    for (const auto &e : active.forces)
      if (t >= e.on - 1e-9 && t < e.off - 1e-9) force += e.force;
    if (force != gz::math::Vector3d::Zero) gz::sim::Link(link).AddWorldForce(ecm, force);
    MoveStreaks(ecm, now, force);
    if (force != lastForce) {
      std::ostringstream s;
      s << "{\"event\":\"force\",\"t\":" << t << ",\"sim_time\":" << now << ",\"force_enu_N\":["
        << force.X() << "," << force.Y() << "," << force.Z() << "]}";
      Write(s.str());
      lastForce = force;
    }
    if (active.detach && !detached && t >= *active.detach - 1e-9) {
      detachPub.Publish(gz::msgs::Empty());
      detached = true;
      std::ostringstream s;
      s << "{\"event\":\"detach\",\"t\":" << t << ",\"sim_time\":" << now << "}";
      Write(s.str());
    }
  }

 private:
  void MoveStreaks(gz::sim::EntityComponentManager &ecm, double now, const gz::math::Vector3d &force) {
    if (streakModels.empty() || now - lastStreakUpdate < .02) return;
    const double dt = lastStreakUpdate < 0 ? 0. : now - lastStreakUpdate;
    lastStreakUpdate = now;
    const gz::math::Vector3d wind(force.X() / dragCoefficient, force.Y() / dragCoefficient, 0.);
    const bool blowing = wind.Length() > .3;
    if (!blowing && streaksParked) return;
    const double yaw = std::atan2(wind.Y(), wind.X());
    const gz::math::Vector3d size = areaMax - areaMin;
    for (size_t k = 0; k < streakModels.size(); ++k) {
      if (streakModels[k] == gz::sim::kNullEntity) {
        streakModels[k] = ecm.EntityByComponents(gz::sim::components::Name(streakPrefix + std::to_string(k)),
                                                 gz::sim::components::Model());
        if (streakModels[k] == gz::sim::kNullEntity) continue;
      }
      auto &p = streakPos[k];
      p += wind * dt;
      p.X(areaMin.X() + std::fmod(std::fmod(p.X() - areaMin.X(), size.X()) + size.X(), size.X()));
      p.Y(areaMin.Y() + std::fmod(std::fmod(p.Y() - areaMin.Y(), size.Y()) + size.Y(), size.Y()));
      // the streak is a cylinder along its z axis: pitch 90 deg lays it along the wind
      const gz::math::Pose3d pose = blowing ? gz::math::Pose3d(p.X(), p.Y(), p.Z(), 0., GZ_PI / 2, yaw)
                                            : gz::math::Pose3d(p.X(), p.Y(), -20., 0., GZ_PI / 2, 0.);
      gz::sim::Model(streakModels[k]).SetWorldPoseCmd(ecm, pose);
    }
    streaksParked = !blowing;
  }

  void Write(const std::string &line) {
    if (log.is_open()) {
      log << line << "\n";
      log.flush();
    }
  }
};
}  // namespace copter_disturbance

GZ_ADD_PLUGIN(copter_disturbance::DisturbanceSystem, gz::sim::System,
              copter_disturbance::DisturbanceSystem::ISystemConfigure,
              copter_disturbance::DisturbanceSystem::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(copter_disturbance::DisturbanceSystem, "copter_disturbance::DisturbanceSystem")
