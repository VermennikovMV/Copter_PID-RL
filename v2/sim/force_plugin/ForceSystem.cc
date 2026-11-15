#include <gz/sim/System.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Name.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/entity_wrench.pb.h>
#include <gz/plugin/Register.hh>
#include <sdf/Element.hh>
#include <fstream>
#include <iomanip>
#include <mutex>

namespace vkr {
class ForceSystem : public gz::sim::System,
                    public gz::sim::ISystemConfigure,
                    public gz::sim::ISystemPreUpdate {
  gz::transport::Node node;
  std::mutex mutex;
  gz::msgs::EntityWrench requested;
  long sequence=-1, appliedSequence=-1;
  std::ofstream log;
 public:
  void Configure(const gz::sim::Entity &,const std::shared_ptr<const sdf::Element> &sdf,
                 gz::sim::EntityComponentManager &,gz::sim::EventManager &) override {
    auto topic=sdf->Get<std::string>("topic");
    log.open(sdf->Get<std::string>("log_path"));
    if(!log.is_open())throw std::runtime_error("Cannot open force application log");
    log<<std::setprecision(17);
    if(!node.Subscribe(topic,&ForceSystem::Receive,this))throw std::runtime_error("Cannot subscribe force command");
  }
  void Receive(const gz::msgs::EntityWrench &msg) {
    long seq=-1;
    for(const auto &field:msg.header().data())
      if(field.key()=="seq" && field.value_size())seq=std::stol(field.value(0));
    std::lock_guard<std::mutex> guard(mutex);
    if(seq>sequence){requested=msg;sequence=seq;}
  }
  void PreUpdate(const gz::sim::UpdateInfo &info,gz::sim::EntityComponentManager &ecm) override {
    if(info.paused)return;
    std::lock_guard<std::mutex> guard(mutex);
    if(sequence<0)return;
    auto entity=static_cast<gz::sim::Entity>(requested.entity().id());
    bool valid=ecm.Component<gz::sim::components::Link>(entity)!=nullptr;
    const auto &f=requested.wrench().force();
    if(valid)gz::sim::Link(entity).AddWorldForce(ecm,{f.x(),f.y(),f.z()});
    if(sequence!=appliedSequence){
      double seconds=std::chrono::duration<double>(info.simTime).count();
      log<<"{\"event\":\"force_physics_step\",\"seq\":"<<sequence
         <<",\"sim_time\":"<<seconds<<",\"iteration\":"<<info.iterations
         <<",\"entity_id\":"<<entity<<",\"link_valid\":"<<(valid?"true":"false")
         <<",\"force_enu_N\":["<<f.x()<<","<<f.y()<<","<<f.z()<<"]}\n";
      log.flush();appliedSequence=sequence;
    }
  }
};
}
GZ_ADD_PLUGIN(vkr::ForceSystem,gz::sim::System,vkr::ForceSystem::ISystemConfigure,vkr::ForceSystem::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(vkr::ForceSystem,"vkr::ForceSystem")
