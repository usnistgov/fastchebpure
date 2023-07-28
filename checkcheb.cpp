// C++ standard library
#include <vector>
#include <valarray>
#include <string>
#include <fstream>
#include <filesystem>
#include <optional>
#include <numeric>
#include <unordered_map>

#include "teqp/models/multifluid.hpp"
#include "teqp/models/multifluid_ancillaries.hpp"
#include "teqp/algorithms/VLE_pure.hpp"
#include "teqp/algorithms/critical_pure.hpp"

// Imports from boost
#include <boost/multiprecision/cpp_bin_float.hpp>
#include "boost/functional/hash.hpp"
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>
using namespace boost::multiprecision;
using my_float_mp = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<100>>;

#include "ChebTools/ChebTools.h"

const std::string teqp_datapath = "../teqp_REFPROP10";

auto check_superancillaries(const std::string& fluid, const std::string& input_file_path, const std::string& outfile) {
    const std::string fluid_json_path = teqp_datapath + "/dev/fluids/" + fluid + ".json";
    auto model = teqp::build_multifluid_model({ fluid_json_path }, teqp_datapath);

    auto get_collection = [](const std::string& expansion_file){
        const nlohmann::json jfile = teqp::load_a_JSON_file(expansion_file);
        
        std::vector<ChebTools::ChebyshevExpansion> oL;
        for (const auto& ex : jfile.at("jexpansionsL")) {
            oL.emplace_back(ex.at("coef").get<std::vector<double>>(), ex.at("xmin"), ex.at("xmax"));
        }
        
        std::vector<ChebTools::ChebyshevExpansion> oV;
        for (const auto& ex : jfile.at("jexpansionsV")) {
            oV.emplace_back(ex.at("coef").get<std::vector<double>>(), ex.at("xmin"), ex.at("xmax"));
        }
        
        return std::make_tuple(ChebTools::ChebyshevCollection(oL), ChebTools::ChebyshevCollection(oV));
    };

    struct FailedIteration : public std::exception {
        std::string msg;
        double T;
        FailedIteration(double T, const std::string& msg) : T(T), msg(msg) {};
    };
    auto db = nlohmann::json::array();

    // Load expansions from file for liquid and vapor
    auto [ccL, ccV] = get_collection(input_file_path);
    auto& ccL_ = ccL, ccV_ = ccV;
    auto meta = teqp::load_a_JSON_file(input_file_path)["meta"];
    double Tcrittrue = meta.at("Tcrittrue / K");
    double rhocrittrue = meta.at("rhocrittrue / mol/m^3");
    
    auto get_degreedoubled_nodes = [&]() {
        std::vector<double> x;
        for (auto cc : { ccL_ }) {
            for (auto& ex : cc.get_exps()) {
                auto N = ex.coef().size() - 1;
                auto nodes_doubled = ChebTools::ChebyshevExpansion::factory(2*N, [](double x) { return x; }, ex.xmin(), ex.xmax()).get_nodes_realworld();
                for (auto& n : nodes_doubled) {
                    x.push_back(n);
                }
            }
        }
        std::sort(x.begin(), x.end());
        return x;
    };

    // Collect all the nodes from the expansions, and their degree-doubled in-between nodes
    std::vector<double> Tnodes = get_degreedoubled_nodes();
    
    for (auto T : Tnodes) {
        try {
            double Theta = (Tcrittrue-T)/Tcrittrue;
            
            double rhoSAL = ccL(T) ;//+ rhocrittrue + meta["BrhoL / mol/m^3"].get<double>()*sqrt(Theta);
            double rhoSAV = ccV(T) ;//+ rhocrittrue + meta["BrhoV / mol/m^3"].get<double>()*sqrt(Theta);
            
            Eigen::ArrayXd rhovec;
            if (std::abs(T - Tcrittrue) > 1e-14 && T < ccL.get_exps().back().xmax() && T < ccV.get_exps().back().xmax()) {
                teqp::IsothermPureVLEResiduals<decltype(model), my_float_mp, teqp::ADBackends::multicomplex> residual(model, T);
                rhovec = do_pure_VLE_T<decltype(residual), my_float_mp>(residual, rhoSAL*(1+1e-5), rhoSAV*(1-1e-5), 10).cast<double>();
//                rhovec = teqp::pure_VLE_T<decltype(model), my_float_mp, teqp::ADBackends::multicomplex>(model, T, rhoSAL*(1+1e-5), rhoSAV*(1-1e-5), 10).cast<double>();
                if (!std::isfinite(rhovec[0])) {
                    throw FailedIteration(T, "Iteration failed @T=" + std::to_string(T) + " K for " + fluid + ". Tcrittrue is " + std::to_string(Tcrittrue) + " K.");
                }
            }
            else {
                rhovec = (Eigen::Array2d() << rhocrittrue, rhocrittrue).finished();
            }
            if (rhovec.size() != 2) {
                std::cout << "rhovec is not 2 elements in length" << std::endl;
            }
            
            db.push_back(nlohmann::json{ 
                {"T / K", T}, 
                {"rho'(SA) / mol/m^3", rhoSAL}, 
                {"rho'(mp) / mol/m^3", rhovec[0]},
                {"rho'(SA)/rho'(mp)", rhoSAL/rhovec[0]},
                {"rho''(SA) / mol/m^3", rhoSAV}, 
                {"rho''(mp) / mol/m^3", rhovec[1]},
                {"rho''(SA)/rho''(mp)", rhoSAV/rhovec[1]},
            });
        }
        catch (FailedIteration& f) {
            if (f.T > 0.9999 * Tcrittrue) {

            }
            else {
                std::cout << f.msg << std::endl;
            }
        }
    }
    
    // Return results
    nlohmann::json jo = {
        {"meta", meta},
        {"data", db}
    };
    std::ofstream ofs(outfile); ofs << jo.dump(2); 
}

const std::string output_prefix = "../output/";
const std::string check_destination = output_prefix + "/check/";
const int FASTCHEB_PROCESSORS = 6;

// Prototype for builder
void build_superancillaries(const std::string &fluid, const std::string &ofpath);

int main(){
    
    // Check for existence of output locations, they must be present
    if (!std::filesystem::exists(output_prefix)){
        std::cout << "output prefix doesn't exist: " << output_prefix << std::endl;
        return EXIT_FAILURE;
    }
    if (!std::filesystem::exists(check_destination)) {
        std::cout << "output checkfile destination doesn't exist: " << check_destination << std::endl;
        return EXIT_FAILURE;
    }

    // Launch the pool with desired number of threads.
    boost::asio::thread_pool pool(FASTCHEB_PROCESSORS);
    
    for (auto const& dir_entry : std::filesystem::directory_iterator{ teqp_datapath + "/dev/fluids" }){
        if (dir_entry.is_regular_file()) {
            auto fluid = dir_entry.path().stem().string();
            
            // Skip the .DS_Store file on OSX
            if (fluid == ".DS_Store"){ continue; }
            
            auto job = [fluid]() {
                auto outfile_path = output_prefix + fluid + "_exps.json";
                auto checkfile_path = output_prefix + "/check/" + fluid + "_check.json";
                
                try {
                    if (!std::filesystem::exists(outfile_path)) {
                        std::cout << "Building ->: " << outfile_path << std::endl;
                        build_superancillaries(fluid, outfile_path);
                    }
                    if (!std::filesystem::exists(checkfile_path)){
                        std::cout << "Checking ->: " << checkfile_path << std::endl;
                        check_superancillaries(fluid, outfile_path, checkfile_path);
                    }
                }
                catch (const std::exception& e) {
                    std::cout << "[" << fluid << "]: " << e.what() << std::endl;
                }
            };

            if (FASTCHEB_PROCESSORS > 0) {
                // Submit a lambda object to the pool.
                std::cout << "Submitting: " << fluid << std::endl;
                boost::asio::post(pool, job);
            }
            else {
                // Run the job serially
                job();
            }
        }
    }
    pool.join();
    return EXIT_SUCCESS;
}
