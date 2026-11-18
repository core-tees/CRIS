#include <native_lm/lm_optimizer.hpp>
#include <iostream>
#include <vector>

void require(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
void near(double a,double b) {if(std::abs(a-b)>=1e-12)throw std::runtime_error("LM value parity: "+std::to_string(a)+" vs "+std::to_string(b));}

void scenario(bool reject_first,bool fail_validation,bool resumed) {
    native_lm::LmState state;
    state.damping=1;state.objective_loss=1;state.training_mse=1;state.validation_mse=1;state.best_validation_mse=1;
    if(resumed) {state.iteration=7;state.accepted_updates=3;}
    native_lm::LmControl control;
    control.max_iterations=3;control.max_accepted_updates=state.accepted_updates+1;
    control.max_validation_failures=fail_validation?1:500;
    int refreshes=0,evaluations=0;
    std::vector<native_lm::LmIteration> events;
    const auto result=native_lm::optimize(0.0,control,state,
        [&](double model) {
            ++refreshes;native_lm::LmLinearization l;
            l.hessian=torch::ones({1,1},torch::kFloat64);
            l.gradient=torch::full({1,1},model-1,torch::kFloat64);l.sample_count=1;return l;
        },
        [](double model,const torch::Tensor& step) {return model+step.item<double>();},
        [&](double model) {
            native_lm::CandidateEvaluation e;++evaluations;
            e.objective_loss=reject_first&&evaluations==1?2:std::pow(model-1,2);
            e.training_mse=e.objective_loss;e.validation_mse=fail_validation?2:e.objective_loss;return e;
        },
        [&](double,const native_lm::LmIteration& e) {events.push_back(e);});
    near(result.model,reject_first?1.0/11:.5);
    require(refreshes==1,"rejected proposals must reuse linearization");
    require(events.size()==(reject_first?2:1),"proposal count");
    require(result.state.accepted_updates==state.accepted_updates+1,"accepted counter");
    require(result.state.iteration==state.iteration+int(events.size()),"resumed iteration counter");
    near(result.state.damping,reject_first?5:.5);
    near(result.state.mean_objective_gradient_norm,2);
    require(result.stopping_reason==(fail_validation?"max_validation_fail":"max_accepted_updates"),"stopping priority");
    require(result.state.validation_failures==(fail_validation?1:0),"validation patience");
    if(reject_first) {
        require(events[0].status==native_lm::LmIteration::Status::Rejected,"rejection event");
        require(!events[1].linearization_refreshed,"cached retry event");
        require(events[0].state.validation_failures==0,"rejected proposal consumed patience");
    }
}

// Tiny deterministic problem: exercise restart state without network training.
// The intentionally worse validation metric makes patience observable.
void restart_contract(bool split_after_rejection) {
    struct Saved {double model;native_lm::LmState state;};
    native_lm::LmState start;
    start.damping=1;start.objective_loss=1;start.training_mse=1;
    start.validation_mse=1;start.best_validation_mse=.01;
    auto run=[&](Saved saved,int proposals,std::vector<native_lm::LmIteration>& events) {
        native_lm::LmControl control;
        control.max_iterations=proposals;control.max_accepted_updates=20;
        control.max_validation_failures=3;
        return native_lm::optimize(saved.model,control,saved.state,
            [](double model) {
                native_lm::LmLinearization l;
                l.hessian=torch::ones({1,1},torch::kFloat64);
                l.gradient=torch::full({1,1},model-1,torch::kFloat64);
                l.sample_count=1;return l;
            },
            [](double model,const torch::Tensor& step){return model+step.item<double>();},
            [=](double model) {
                native_lm::CandidateEvaluation e;
                // A model-based rejection remains deterministic across restart.
                e.objective_loss=split_after_rejection&&model==.5?2:std::pow(model-1,2);
                e.training_mse=e.objective_loss;e.validation_mse=2;return e;
            },
            [&](double,const native_lm::LmIteration& e){events.push_back(e);});
    };
    std::vector<native_lm::LmIteration> full_events,split_events;
    const auto full=run({0,start},20,full_events);
    const auto first=run({0,start},split_after_rejection?1:2,split_events);
    const auto resumed=run({first.model,first.state},20-first.state.iteration,split_events);
    near(full.model,resumed.model);
    require(full.stopping_reason==resumed.stopping_reason,"restart stopping reason");
    require(full_events.size()==split_events.size(),"restart proposal count");
    for(std::size_t i=0;i<full_events.size();++i) {
        const auto& a=full_events[i];const auto& b=split_events[i];
        require(a.status==b.status,"restart accepted/rejected sequence");
        require(a.state.iteration==b.state.iteration,"restart iteration offset");
        require(a.state.accepted_updates==b.state.accepted_updates,"restart accepted count");
        require(a.state.validation_failures==b.state.validation_failures,"restart validation patience");
        near(a.state.damping,b.state.damping);
        near(a.state.objective_loss,b.state.objective_loss);
        near(a.state.best_validation_mse,b.state.best_validation_mse);
    }
    if(!split_after_rejection) {
        auto lost_patience=first.state;lost_patience.validation_failures=0;
        std::vector<native_lm::LmIteration> reset_events;
        const auto reset=run({first.model,lost_patience},20-first.state.iteration,reset_events);
        require(reset.state.accepted_updates==full.state.accepted_updates+2,
            "resetting patience must expose the two extra accepted updates");
    }
}

void exhausted_budget() {
    for(bool patience:{false,true}) {
        native_lm::LmState state; state.iteration=7;state.accepted_updates=3;
        state.validation_failures=patience?3:0;
        native_lm::LmControl control;control.max_iterations=10;
        control.max_accepted_updates=3;control.max_validation_failures=3;
        const auto result=native_lm::optimize(2.0,control,state,
            [](double)->native_lm::LmLinearization{throw std::runtime_error("Exhausted run linearized");},
            [](double,const torch::Tensor&)->double{throw std::runtime_error("Exhausted run proposed");},
            [](double)->native_lm::CandidateEvaluation{throw std::runtime_error("Exhausted run evaluated");},
            [](double,const native_lm::LmIteration&){throw std::runtime_error("Exhausted run emitted a step");});
        require(result.state.iteration==7 && result.state.accepted_updates==3,"exhausted counters changed");
        require(result.stopping_reason==(patience?"max_validation_fail":"max_accepted_updates"),"exhausted stop reason");
    }
}

void scaled_trial_steps() {
    native_lm::LmState state;
    state.damping=1;state.objective_loss=1;state.training_mse=1;
    state.validation_mse=1;state.best_validation_mse=1;
    native_lm::LmControl control;
    control.max_iterations=1;control.scaled_cholesky=true;
    control.max_damping_trials=2;control.max_step_trials=3;
    int evaluations=0;
    const auto result=native_lm::optimize(0.0,control,state,
        [](double model) {
            native_lm::LmLinearization l;
            l.hessian=torch::ones({1,1},torch::kFloat64);
            l.gradient=torch::full({1,1},model-1,torch::kFloat64);
            l.sample_count=1;return l;
        },
        [](double model,const torch::Tensor& step){return model+step.item<double>();},
        [&](double model) {
            ++evaluations;native_lm::CandidateEvaluation e;
            e.objective_loss=model>.49?2:std::pow(model-1,2);
            e.training_mse=e.objective_loss;e.validation_mse=e.objective_loss;return e;
        },
        [](double,const native_lm::LmIteration&){});
    near(result.model,.25);
    require(evaluations==2,"scaled step backtracking");
    require(result.state.accepted_updates==1,"scaled accepted count");
}

int main() {
    try {
        at::set_num_threads(1);
        scenario(false,false,false);scenario(true,false,false);
        scenario(false,true,false);scenario(false,false,true);
        restart_contract(false);restart_contract(true);
        exhausted_budget();
        scaled_trial_steps();
        std::cout<<"PASS: LM step, damping, rejected-step reuse, patience, accepted/resumed counters\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
