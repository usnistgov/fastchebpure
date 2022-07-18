// C++ standard library
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <optional>
#include <unordered_map>

#include "teqp/models/multifluid.hpp"
#include "teqp/models/multifluid_ancillaries.hpp"
#include "teqp/algorithms/VLE.hpp"

// Imports from boost
#include <boost/multiprecision/cpp_bin_float.hpp>
#include "boost/functional/hash.hpp"
using namespace boost::multiprecision;

#include "ChebTools/ChebTools.h"

const std::string teqp_datapath = "../externals/teqp/mycp";

template<int ifluid>
auto build_superancillaries(const std::string &fluid, const std::optional<std::string> &ofsuffix){
    const std::string fluid_json_path = teqp_datapath + "/dev/fluids/" + fluid + ".json";
    auto model = teqp::build_multifluid_model({ fluid_json_path }, teqp_datapath);

    // Build conventional ancillaries
    auto build_ancillaries = [](const auto& c) {
        if (c.redfunc.Tc.size() != 1) {
            throw teqp::InvalidArgument("Can only build ancillaries for pure fluids");
        }
        auto jancillaries = nlohmann::json::parse(c.get_meta()).at("pures")[0].at("ANCILLARIES");
        return teqp::MultiFluidVLEAncillaries(jancillaries);
    };
    auto anc = build_ancillaries(model); 

    // The calculation cache
    struct DensitiesType { 
        const double rhoL, rhoV; 
        DensitiesType(double rhoL, double rhoV) : rhoL(rhoL), rhoV(rhoV) {}; 
    };
    struct KeyHash {
        std::size_t operator()(const double& key) const {
            return boost::hash_value(key);
        }
    };
    std::unordered_map<double, DensitiesType, KeyHash > densitydb;

    auto get_densities = [&anc, &model, &densitydb](double T) {
        if (densitydb.count(T) == 0) { // If not in cache...
            // Do the calculation and store in cache
            using my_float_mp = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<100>>;
            auto rhovec = teqp::pure_VLE_T<decltype(model), my_float_mp, teqp::ADBackends::multicomplex>(model, T, anc.rhoL(T), anc.rhoV(T), 10).cast<double>();
            densitydb.insert(std::make_pair(T, DensitiesType(rhovec[0], rhovec[1])));
        }
        auto d = densitydb.at(T); // Retrieve from cache
        return std::make_tuple(d.rhoL, d.rhoV);
    };
    auto j = teqp::load_a_JSON_file(fluid_json_path);
    double Tcrit = j.at("STATES").at("critical").at("T"); // Critical temperature
    double Treducing = j.at("EOS")[0].at("STATES").at("reducing").at("T"); // Reducing temperature 
    double Ttriple = j.at("STATES").at("triple_liquid").at("T"); // Triple-point (SLV) temperature
    double R = j.at("EOS")[0].at("gas_constant"); // Gas constant being used

    double Tmin = Ttriple, Tmax = std::min(Treducing, Tcrit), tol = 1e-12;
    int N = 12, Msplit = 3, max_refine_passes = 12;
    std::function<void(int, const std::deque<ChebTools::ChebyshevExpansion>&)> callback = [](int num_pass, const std::deque<ChebTools::ChebyshevExpansion>& exs) {
        std::cout << num_pass << std::endl;
    };
    auto exps = ChebTools::ChebyshevExpansion::dyadic_splitting(
        N, 
        [&get_densities](double T) { return std::get<ifluid>(get_densities(T)); }, 
        Tmin, Tmax, Msplit, tol, max_refine_passes, callback
    );
    nlohmann::json meta = {
        {"Tcrit", Tcrit},
        { "Treducing", Treducing },
        { "Ttriple", Ttriple },
        {"gas_constant", R}
    };
    
    if (ofsuffix.has_value()){
        nlohmann::json jexpansions = nlohmann::json::array();
        auto tovec = [](const Eigen::ArrayXd& a) {
            std::vector<double> z(a.size());
            for (auto i = 0; i < a.size(); ++i) { z[i] = a[i]; }
            return z;
        };
        for (auto& ex : exps) {
            jexpansions.push_back({
                {"coef", tovec(ex.coef())},
                {"xmin", ex.xmin()}, 
                {"xmax", ex.xmax()},
            });
        }
        nlohmann::json jo = {
            {"meta", meta},
            {"jexpansions", jexpansions}
        };
        std::ofstream ofs(fluid + ofsuffix.value()); ofs << jo.dump(2); 
    }
    return std::make_tuple(exps, meta);
}

int main(){
    for (auto const& dir_entry : std::filesystem::directory_iterator{ teqp_datapath + "/dev/fluids" }){
        if (dir_entry.is_regular_file()) {
            auto fluid = std::filesystem::absolute(dir_entry).filename().stem().string();
            std::cout << fluid << '\n';
            auto expsL = build_superancillaries<0>(fluid, "_expsL.json");
            auto expsV = build_superancillaries<1>(fluid, "_expsV.json");
        }
    }
    return EXIT_SUCCESS;
}