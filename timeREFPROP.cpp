
// Only this file gets the implementation
#define REFPROP_IMPLEMENTATION
#define REFPROP_FUNCTION_MODIFIER
#include "REFPROP_lib.h"
#undef REFPROP_FUNCTION_MODIFIER
#undef REFPROP_IMPLEMENTATION

#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <valarray>
#include <random>

int main()
{
    // You may need to change this path to suit your installation
    // Note: forward-slashes are recommended.
    std::string path = "C:/Program Files (x86)/REFPROP";
    std::string DLL_name = "REFPRP64.dll";

    // Load the shared library
    std::string err;
    
    bool loaded_REFPROP = load_REFPROP(err, path, DLL_name);
    printf("Loaded refprop (in main.cpp): %s @ address %zu\n", loaded_REFPROP ? "true" : "false", REFPROP_address());
    if (!loaded_REFPROP){
        return EXIT_FAILURE;
    }
 
    SETPATHdll(const_cast<char*>(path.c_str()), 400);

    int ierr = 0, nc = 1;
    char herr[255], hfld[10000] = "PROPANE", hhmx[255] = "HMX.BNC", href[4] = "DEF";
    SETUPdll(nc,hfld,hhmx,href,ierr,herr,10000,255,3,255);

    if (ierr > 0) printf("This ierr: %d herr: %s\n", ierr, herr);
    {
        int ierr = 0, kq = 1;
        char herr[255];
        double z[20] = {1.0}, x[20] = {1.0}, y[20] = {1.0}, T= 300, p = 101.325, d = -1, dl = -1, dv = -1, h = -1, s = -1, u = -1, cp = -1, cv = -1, q = 0, w = -1;

        // Random speed testing
        std::uniform_real_distribution<double> unif(89, 360);
        std::default_random_engine re;
        double a_random_double = unif(re);

        std::valarray<double> Ts(100000); std::transform(std::begin(Ts), std::end(Ts), std::begin(Ts), [&unif, &re](double x) { return unif(re); });
        std::valarray<double> ps = 0.0 * Ts;
        int kph = 1;
        for (auto repeat = 0; repeat < 10; ++repeat) {
            auto tic = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < Ts.size(); ++i) {
                SATTdll(Ts[i], z, kph, p, dl, dv, x, y, ierr, herr, 255);
                ps[i] = p;
            }
            auto toc = std::chrono::high_resolution_clock::now();
            double elap_us = std::chrono::duration<double>(toc - tic).count() / Ts.size() * 1e6;
            std::cout << elap_us << " microseconds per call for SATTdll" << std::endl;
        }
    }
    {
        int ierr = 0, kq = 1;
        char herr[255];
        double z[20] = {1.0}, x[20] = {1.0}, y[20] = {1.0}, T= 300, d = -1, dl = -1, dv = -1, h = -1, s = -1, u = -1, cp = -1, cv = -1, q = 0, w = -1;

        // Random speed testing
        double pmin = 1e-4, pmax = 4251.2;
        std::uniform_real_distribution<double> unif(log(pmin), log(pmax));
        std::default_random_engine re;
        double a_random_double = unif(re);

        std::valarray<double> lnp(100000); std::transform(std::begin(lnp), std::end(lnp), std::begin(lnp), [&unif, &re](double x) { return unif(re); });
        std::valarray<double> ps = exp(lnp), Ts = 0.0*ps;

        for (auto repeat = 0; repeat < 10; ++repeat) {

            auto tic = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < ps.size(); ++i) {
                PQFLSHdll(ps[i], q, z, kq, T, d, dl, dv, x, y, u, h, s, cv, cp, w, ierr, herr, 255);
                Ts[i] = T;
            }
            auto toc = std::chrono::high_resolution_clock::now();
            double elap_us = std::chrono::duration<double>(toc - tic).count() / Ts.size() * 1e6;
            std::cout << elap_us << " microseconds per call for PQFLSHdll" << std::endl;
        }
    }
    return EXIT_SUCCESS;
}