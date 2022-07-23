// C++ standard library
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <optional>
#include <numeric>
#include <unordered_map>

#include "teqp/models/multifluid.hpp"
#include "teqp/models/multifluid_ancillaries.hpp"
#include "teqp/algorithms/VLE.hpp"

// Imports from boost
#include <boost/multiprecision/cpp_bin_float.hpp>
#include "boost/functional/hash.hpp"
using namespace boost::multiprecision;

#include "ChebTools/ChebTools.h"

const std::string teqp_datapath = "../teqp_backup";
const std::string output_prefix = "../output/";

template<int ifluid>
auto build_superancillaries(const std::string &fluid, const std::optional<std::string> &ofsuffix, const std::optional<std::string>& ofprefix){
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

    // The calculation cache for results of calculations
    struct DensitiesType { const double rhoL, rhoV; };
    std::unordered_map<double, DensitiesType, boost::hash<double>> densitydb;

    auto j = teqp::load_a_JSON_file(fluid_json_path);
    double Tcrit = j.at("STATES").at("critical").at("T"); // Critical temperature
    double Treducing = j.at("EOS")[0].at("STATES").at("reducing").at("T"); // Reducing temperature 
    double rhomolar_reducing = j.at("EOS")[0].at("STATES").at("reducing").at("rhomolar"); // Reducing molar density
    double Ttriple = j.at("STATES").at("triple_liquid").at("T"); // Triple-point (SLV) temperature
    bool pseudo_pure = j.at("EOS")[0].at("pseudo_pure");

    if (pseudo_pure) {
        return std::make_tuple(std::deque<ChebTools::ChebyshevExpansion>{}, nlohmann::json{});
    }
    struct FailedIteration : public std::exception {
        std::string msg;
        double T;
        FailedIteration(double T, const std::string& msg) : T(T), msg(msg) {};
    };

    double R = j.at("EOS")[0].at("gas_constant"); // Gas constant being used
    auto get_densities = [&anc, &model, &densitydb, &Treducing, &rhomolar_reducing](double T) {
        if (std::abs(T / Treducing - 1) < 1e-14) {
            return std::make_tuple(rhomolar_reducing, rhomolar_reducing);
        }
        else if (densitydb.count(T) == 0) { // If not in cache...
            // Do the calculation and store in cache
            using my_float_mp = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<100>>;
            auto rhovec = teqp::pure_VLE_T<decltype(model), my_float_mp, teqp::ADBackends::multicomplex>(model, T, anc.rhoL(T), anc.rhoV(T), 10).cast<double>();
            if (!std::isfinite(rhovec[0])) {
                throw FailedIteration(T, "Iteration failed @T="+std::to_string(T)+". Treducing is "+std::to_string(Treducing));
            }
            densitydb.insert(std::make_pair(T, DensitiesType{ rhovec[0], rhovec[1] }));
        }
        auto d = densitydb.at(T); // Retrieve from cache
        return std::make_tuple(d.rhoL, d.rhoV);
    };

    double Tmin = Ttriple, Tmax = std::min(Treducing, Tcrit), tol = 1e-12;
    int N = 12, Msplit = 3, max_refine_passes = 12;
    std::deque<ChebTools::ChebyshevExpansion> last_good_exs;
    std::function<void(int, const std::deque<ChebTools::ChebyshevExpansion>&)> callback = [&last_good_exs](int num_pass, const std::deque<ChebTools::ChebyshevExpansion>& exs) {
        std::cout << ".";
        last_good_exs = exs;
    };
    decltype(last_good_exs) exps;
    try {
        exps = ChebTools::ChebyshevExpansion::dyadic_splitting(
            N,
            [&get_densities](double T) { return std::get<ifluid>(get_densities(T)); },
            Tmin, Tmax, Msplit, tol, max_refine_passes, callback
        );
    }
    catch (FailedIteration&f) {
        if (f.T > 0.9999*Tmax){
            exps = last_good_exs;
        }
        else {
            throw;
        }
    }
    std::cout << std::endl;
    nlohmann::json meta = {
        { "Tcrit", Tcrit },
        { "Treducing", Treducing },
        { "Ttriple", Ttriple },
        { "gas_constant", R }
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
        std::string prefix = ofprefix.value_or("");
        std::ofstream ofs(prefix + fluid + ofsuffix.value()); ofs << jo.dump(2); 
    }
    return std::make_tuple(exps, meta);
}

int main(){
    for (auto const& dir_entry : std::filesystem::directory_iterator{ teqp_datapath + "/dev/fluids" }){
        if (dir_entry.is_regular_file()) {
            auto fluid = dir_entry.path().stem().string();
            std::string ofL = output_prefix + fluid + "_expsL.json";
            if (std::filesystem::exists(ofL)) {
                continue;
            }
            std::cout << fluid << '\n';
            try {
                auto expsL = build_superancillaries<0>(fluid, "_expsL.json", output_prefix);
                auto expsV = build_superancillaries<1>(fluid, "_expsV.json", output_prefix);
            }
            catch (const std::exception &e) {
                std::cout << e.what() << std::endl;
            }
        }
    }
    return EXIT_SUCCESS;
}