#define CATCH_CONFIG_ENABLE_BENCHMARKING
#define CATCH_CONFIG_MAIN
#include <catch/catch.hpp>

#include <stdio.h>
#include <string>

#include "ChebTools/ChebTools.h"
using namespace ChebTools;

#include "teqp/json_tools.hpp"

// Only this file gets the implementation
#define REFPROP_IMPLEMENTATION
#define REFPROP_FUNCTION_MODIFIER
#include "REFPROP_lib.h"
#undef REFPROP_FUNCTION_MODIFIER
#undef REFPROP_IMPLEMENTATION

TEST_CASE("Time water superancillary", "[water]")
{
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
    const std::string output_prefix = "../output/";
    auto exps = get_collection(output_prefix + "/WATER_exps.json");
    
    BENCHMARK("450 K") {
        return std::get<0>(exps)(450);
    };
    BENCHMARK("275 K") {
        return std::get<0>(exps)(275);
    };
    BENCHMARK("540 K") {
        return std::get<0>(exps)(540);
    };
}

TEST_CASE("Time water w/ REFPROP", "[water]")
{
    
    // You may need to change this path to suit your installation
    // Note: forward-slashes are recommended.
    std::string path = getenv("HOME") + std::string("/REFPROP10");
    std::string DLL_name = "librefprop.dylib";

    // Load the shared library
    std::string err;
    
    bool loaded_REFPROP = load_REFPROP(err, path, DLL_name);
    printf("Loaded refprop (in main.cpp): %s @ address %zu\n", loaded_REFPROP ? "true" : "false", REFPROP_address());
    if (!loaded_REFPROP){
        return EXIT_FAILURE;
    }
    
    {
        char hFlag[255] = "Cache   ";
        int jFlag = 3, kFlag = -1000;
        int ierr = 0; char herr[255];
        FLAGSdll(hFlag, jFlag, kFlag, ierr, herr, 255, 255);
    }
 
    SETPATHdll(const_cast<char*>(path.c_str()), 400);

    int ierr = 0, nc = 1;
    char herr[255], hfld[10000] = "WATER", hhmx[255] = "HMX.BNC", href[4] = "DEF";
    SETUPdll(nc,hfld,hhmx,href,ierr,herr,10000,255,3,255);
    
    int kq = 1;
    double z[20] = {1.0}, x[20] = {1.0}, y[20] = {1.0}, T= 300, p = 101.325, d = -1, dl = -1, dv = -1, h = -1, s = -1, u = -1, cp = -1, cv = -1, q = 0, w = -1;
    int kph = 1;
    
    BENCHMARK("275 K") {
        double T = 275;
        SATTdll(T, z, kph, p, dl, dv, x, y, ierr, herr, 255);
        return p;
    };
    BENCHMARK("300 K") {
        double T = 300;
        SATTdll(T, z, kph, p, dl, dv, x, y, ierr, herr, 255);
        return p;
    };
    BENCHMARK("325 K") {
        double T = 325;
        SATTdll(T, z, kph, p, dl, dv, x, y, ierr, herr, 255);
        return p;
    };
    BENCHMARK("350 K") {
        double T = 350;
        SATTdll(T, z, kph, p, dl, dv, x, y, ierr, herr, 255);
        return p;
    };
    BENCHMARK("450 K") {
        double T = 450;
        SATTdll(T, z, kph, p, dl, dv, x, y, ierr, herr, 255);
        return p;
    };
    BENCHMARK("540 K") {
        double T = 540;
        SATTdll(T, z, kph, p, dl, dv, x, y, ierr, herr, 255);
        return p;
    };
}
