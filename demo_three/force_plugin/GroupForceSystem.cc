// A single physical update applies the same world force to three distinct links.
// This demo-only plugin leaves the original research ForceSystem untouched.
#include <gz/sim/System.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/components/Link.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/entity_wrench.pb.h>
#include <gz/plugin/Register.hh>
#include <sdf/Element.hh>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <stdexcept>

namespace demo_three {
class GroupForceSystem : public gz::sim::System,
                         public gz::sim::ISystemConfigure,
                         public gz::sim::ISystemPreUpdate {
  gz::transport::Node node;
  std::mutex mutex;
  std::array<gz::sim::Entity, 3> entities{};
  std::array<std::ofstream, 3> logs;
  gz::math::Vector3d force{0, 0, 0};
  long sequence = -1, appliedSequence = -1;
 public:
  void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &sdf,
                 gz::sim::EntityComponentManager &, gz::sim::EventManager &) override {
    const auto folder = std::filesystem::path(sdf->Get<std::string>("log_dir"));
    const std::array<std::string, 3> names{"iris_pid", "iris_hybrid", "iris_direct"};
    for (size_t i = 0; i < names.size(); ++i) {
      logs[i].open(folder / ("force_physics_" + names[i] + ".jsonl"));
      if (!logs[i].is_open()) throw std::runtime_error("Cannot open physical force log");
      logs[i] << std::setprecision(17);
    }
    if (!node.Subscribe(sdf->Get<std::string>("topic"), &GroupForceSystem::Receive, this))
      throw std::runtime_error("Cannot subscribe atomic three-drone force command");
  }
  void Receive(const gz::msgs::EntityWrench &message) {
    long seq = -1;
    std::array<gz::sim::Entity, 3> ids{};
    bool idsFound = false;
    try {
      for (const auto &field : message.header().data()) {
        if (field.key() == "seq" && field.value_size() == 1) seq = std::stol(field.value(0));
        if (field.key() == "entity_ids" && field.value_size() == 3) {
          for (int i = 0; i < 3; ++i) ids[i] = std::stoull(field.value(i));
          idsFound = true;
        }
      }
    } catch (const std::exception &) { return; }
    const auto &f = message.wrench().force();
    if (!idsFound || seq < 0 || !ids[0] || !ids[1] || !ids[2] ||
        ids[0] == ids[1] || ids[0] == ids[2] || ids[1] == ids[2] ||
        !std::isfinite(f.x()) || !std::isfinite(f.y()) || !std::isfinite(f.z())) return;
    std::lock_guard<std::mutex> guard(mutex);
    if (seq > sequence) {
      sequence = seq;
      entities = ids;
      force.Set(f.x(), f.y(), f.z());
    }
  }
  void PreUpdate(const gz::sim::UpdateInfo &info, gz::sim::EntityComponentManager &ecm) override {
    if (info.paused) return;
    std::lock_guard<std::mutex> guard(mutex);
    if (sequence < 0) return;
    bool allValid = true;
    for (auto entity : entities)
      allValid = allValid && ecm.Component<gz::sim::components::Link>(entity) != nullptr;
    // Avoid partial delivery if one model/link could not be resolved correctly.
    if (allValid)
      for (auto entity : entities) gz::sim::Link(entity).AddWorldForce(ecm, force);
    if (sequence != appliedSequence) {
      const double seconds = std::chrono::duration<double>(info.simTime).count();
      for (size_t i = 0; i < entities.size(); ++i) {
        logs[i] << "{\"event\":\"force_physics_step\",\"seq\":" << sequence
                << ",\"sim_time\":" << seconds << ",\"iteration\":" << info.iterations
                << ",\"entity_id\":" << entities[i]
                << ",\"link_valid\":" << (allValid ? "true" : "false")
                << ",\"group_atomic\":true,\"force_enu_N\":[" << force.X() << ","
                << force.Y() << "," << force.Z() << "]}\n";
        logs[i].flush();
      }
      appliedSequence = sequence;
    }
  }
};
}
GZ_ADD_PLUGIN(demo_three::GroupForceSystem, gz::sim::System,
              demo_three::GroupForceSystem::ISystemConfigure,
              demo_three::GroupForceSystem::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(demo_three::GroupForceSystem, "demo_three::GroupForceSystem")
