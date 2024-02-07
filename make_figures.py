import os, timeit
from dataclasses import dataclass
from typing import List, Dict

os.environ['RPPREFIX'] = os.getenv('HOME') + '/REFPROP10'

import glob, json, os

import pandas, numpy as np, matplotlib.pyplot as plt
plt.style.use('classic')
plt.style.use('mystyle.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages

import teqp
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary

import ChebTools 
from pypdf import PdfReader
import scipy.optimize 

import teqp

# Set to True to plot only a few, for testing purposes
DEV = False 

def plot_criticals_FLD(*, FLD, Thetamin=1e-6, Thetamax=-1e-6, deltamin=0.9, deltamax=1.1):
    model = teqp.build_multifluid_model([FLD], 'teqp_REFPROP10')
    j = json.load(open(f'output/check/{FLD}_check.json'))
    df = pandas.DataFrame(j['data'])
    df.info()
    plt.plot(df["rho'(mp) / mol/m^3"], df['T / K'], 'k.')
    plt.plot(df["rho''(mp) / mol/m^3"], df['T / K'], 'k.')

    plt.plot(0.5*df["rho''(mp) / mol/m^3"]+0.5*df["rho'(mp) / mol/m^3"], df['T / K'], 'r.')

    Tcrit =  model.get_Tcvec()[0]
    rhomolarcrit = 1/model.get_vcvec()[0]
    Tbox = 0.9*Tcrit, 2*Tcrit
    rhobox = 0.5*rhomolarcrit, 2*rhomolarcrit

    for T in np.linspace(*Tbox):
        for rho in np.linspace(*rhobox):
            Tcrittrue, rhocrittrue = model.solve_pure_critical(T, rho)
            molefrac = np.array([1.0])
            R = model.get_R(molefrac)
            _, Ar01, Ar02 = model.get_Ar02n(Tcrittrue, rhocrittrue, molefrac)
            dpdrho = R*Tcrittrue*(1 + 2*Ar01 + Ar02)
            d2pdrho2 = R*Tcrittrue/(rhocrittrue)*(2*model.get_Ar01(Tcrittrue, rhocrittrue, molefrac) + 4*model.get_Ar02(Tcrittrue, rhocrittrue, molefrac)+model.get_Ar03(Tcrittrue, rhocrittrue, molefrac))
            if abs(dpdrho) > 1e-9: continue
            if abs(d2pdrho2) > 1e-9: continue
            print(Tcrittrue, rhocrittrue, dpdrho, d2pdrho2)
            plt.plot(rhocrittrue, Tcrittrue, 'bs')
    plt.xlim(deltamin*rhocrittrue, deltamax*rhocrittrue)
    plt.ylim((1-Thetamin)*Tcrittrue, (1-Thetamax)*Tcrittrue)
    plt.gca().set(xlabel=r'$\rho$ / mol/m$^3$', ylabel='$T$ / K')

    plt.plot(rhomolarcrit, Tcrit, '*', color='yellow', ms=12)
    plt.tight_layout(pad=0.2)
    plt.savefig(f'{FLD}_near_crit.pdf')
    plt.close()

def numprofile_stats():
    N = []
    for f in sorted(glob.glob('output/*_exps.json')):
        jL = json.load(open(f))['jexpansions_rhoL']
        N.append(len(jL))
    print(np.std(N), np.mean(N), 'std and mean of len of expansions per fluid')
numprofile_stats()
# quit()

def get_expansions(FLD, *, and_p=False):
    """ 
    Return a tuple containing the ChebyshevCollection for the two density superancillary functions
    """
    expsL, expsV = [], []
    for jL in json.load(open(f'output/{FLD}_exps.json'))['jexpansions_rhoL']:
        eL = ChebTools.ChebyshevExpansion(jL['coef'], jL['xmin'], jL['xmax'])
        expsL.append(eL)
    for jV in json.load(open(f'output/{FLD}_exps.json'))['jexpansions_rhoV']:
        eV = ChebTools.ChebyshevExpansion(jV['coef'], jV['xmin'], jV['xmax'])
        expsV.append(eV)

    ceL = ChebTools.ChebyshevCollection(expsL)
    ceV = ChebTools.ChebyshevCollection(expsV)
    if and_p:
        exps = []
        for jV in json.load(open(f'output/{FLD}_exps.json'))['jexpansions_p']:
            eV = ChebTools.ChebyshevExpansion(jV['coef'], jV['xmin'], jV['xmax'])
            exps.append(eV)
        cep = ChebTools.ChebyshevCollection(exps)
        return ceL, ceV, cep
    else:

        return ceL, ceV
    
def dois2bibs(dois):
    import requests 
    import re
    def doi2bib(doi):
        """
        Return a bibTeX string of metadata for a given DOI. See https://gist.github.com/jrsmith3/5513926
        """
        headers = {"accept": "application/x-bibtex"}
        r = requests.get("http://dx.doi.org/" + doi, headers = headers)
        r.encoding = r.apparent_encoding
        if r.ok:
            s = re.sub(r'article{([\S\_]+),', 'article{' + doi + ',', r.text)
            s = re.sub(r'techreport{([\S\_]+),', 'article{' + doi + ',', s)
            return s
        else:
            print("Couldn't get this DOI: {doi}")
            return '?'
        
    print('getting the bibtex for the DOI, be patient...')
    s = '\n\n'.join([doi2bib(doi) for doi in dois if doi])
    s = s.replace(r'{\&}amp$\mathsemicolon$', 'and').replace(r'$\less$/i$\greater$','$')
    s = s.replace(r'$\less$i$\greater$','$').replace(r'{\textendash}','-')
    s = s.replace(r'$\less$b$\greater$$\uprho$$\less$/b$\greater$',r'$\rho$').replace(r'\uprho',r'\rho')
    return s

@dataclass
class ReferenceInfo:
    FLDs: List[str]
    dois: List[str]
    newdois: List[str] # Published since REFPROP 10.0 came out
    FLD_use_CoolProp: List[str]
    keymap: Dict[str, str]

def get_all_references(FLDs):
    dois = []
    newdois = []
    keymap = {}
    FLD_use_CoolProp = ['R114','R113','METHANOL','R12','RC318','R14','FLUORINE','R124','R21','CYCLOPRO','PROPYNE']

    for FLD in FLDs:
        import CoolProp.CoolProp as CP
        if FLD in FLD_use_CoolProp:
            keymap[FLD] = CP.get_fluid_param_string(FLD, 'BibTeX-EOS')
            continue 
        DOI = RP.REFPROPdll(FLD, '','DOI_EOS',0,0,0,0,0,[1.0]).hUnits
        if not DOI:
            hcite = RP.GETMODdll(1,'EOS').hcite
            sandbox = os.getenv('HOME') +  '/Documents/Code/REFPROP-sandbox'
            DOI = RP.REFPROPdll(sandbox+'/FLUIDS/'+FLD+'.FLD', '','DOI_EOS',0,0,0,0,0,[1.0]).hUnits
            if DOI: 
                newdois.append(DOI)
                keymap[FLD] = DOI
            else:
                print(hcite)
        else:
            keymap[FLD] = DOI
            dois.append(DOI)
    
    return ReferenceInfo(dois=dois, newdois=newdois, FLD_use_CoolProp=FLD_use_CoolProp, FLDs=FLDs, keymap=keymap)

def plot_water_nonmono(RP):

    FLD = 'WATER'
    ceL, ceV, cep = get_expansions(FLD=FLD, and_p=True)
    fig, ax = plt.subplots(1,1,figsize=(5.0, 3))
    j = json.load(open(f'output/check/{FLD}_check.json'))
    Tcrit = j['meta']['Tcrittrue / K']
    df = pandas.DataFrame(j['data'])
    df['errL'] = np.abs(df["rho'(SA) / mol/m^3"]/df["rho'(mp) / mol/m^3"]-1)
    df['errV'] = np.abs(df["rho''(SA) / mol/m^3"]/df["rho''(mp) / mol/m^3"]-1)
    df['Theta'] = (Tcrit-df['T / K'])/Tcrit
    plt.plot(df['T / K'], df["rho'(mp) / mol/m^3"], 'o')
    
    Tstationary = []
    for ce in ceL.get_exps():
        Tstationary += ce.deriv(1).real_roots(True)
    Trhomax = Tstationary[0]
    print(Trhomax-273.15, '°C at density maximum')
    x = np.linspace(273.17, 285, 10000)
    y = [ceL(x_) for x_ in x]
    plt.plot(x, y)
    plt.xlim(273.15, 282)
    plt.ylim(55495, 55506)
    # plt.axvline(Trhomax)

    n = 4.0
    def get_T(rt4rho):
        if rt4rho <= rhomin**(1/n): return Tmin
        if rt4rho >= rhomax**(1/n): return Tmax
        rho = rt4rho**n
        def objective(T):
            return (ceL(T)-rho)/rho
        fL = objective(Tmin)
        fR = objective(Tmax)
        if abs(fL) < 1e-10:
            return Tmin
        if abs(fR) < 1e-10:
            return Tmax
        return scipy.optimize.brentq(objective, Tmin, Tmax)
    def callback(i, exps):
        # print(i)
        pass

    Tmin = 273.18
    Tmax = Trhomax
    rhomin = ceL(Tmin)
    rhomax = ceL(Tmax)
    invrt4rhoBrent = ChebTools.ChebyshevCollection(ChebTools.dyadic_splitting(12, get_T, rhomin**(1/n), rhomax**(1/n), 3, 1e-13, 20))
    for rho in np.linspace(rhomin, rhomax, 10000):
        plt.plot(invrt4rhoBrent(rho**0.25), rho, 'c.', mfc=None, zorder=-100)
    
    n = 4.0
    def get_T(rt4rho):
        if rt4rho <= rhomin**(1/n): return Tmax
        if rt4rho >= rhomax**(1/n): return Tmin
        rho = rt4rho**n
        def objective(T):
            return (ceL(T)-rho)/rho
        fL = objective(Tmin)
        fR = objective(Tmax)
        if abs(fL) < 1e-10:
            return Tmin
        if abs(fR) < 1e-10:
            return Tmax
        return scipy.optimize.brentq(objective, Tmin, Tmax)
    
    Tmin = Trhomax
    Tmax = 285
    rhomin = ceL(Tmax)
    rhomax = ceL(Tmin)
    invrt4rhoBrent = ChebTools.ChebyshevCollection(ChebTools.dyadic_splitting(12, get_T, rhomin**(1/n), rhomax**(1/n), 3, 1e-13, 20))
    for rho in np.linspace(rhomin, rhomax, 10000):
        plt.plot(invrt4rhoBrent(rho**0.25), rho, 'c.', mfc=None, zorder=-100)

    RP.SETFLUIDSdll(FLD)
    for rho in np.linspace(55470, np.max(df["rho'(mp) / mol/m^3"]), 200):
        r = RP.REFPROPdll('', 'DSAT','T',RP.MOLAR_BASE_SI, 0, 0,rho, 0,[1.0])
        if r.ierr != 0:
            print(r.herr)
        plt.plot(r.Output[0], rho, 'kX', mfc=None)

    # Idea from: https://stackoverflow.com/a/10517481
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    new_tick_locations = np.array([1, 2, 3, 4, 5, 6, 7, 8])+273.15 # temperatures in K
    def tick_function(X):
        V = X-273.15
        return ["%.0f" % z for z in V]
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations))
    ax2.set_xlabel(r"$t$ / °C")

    ax1.set(xlabel='$T$ / K', ylabel=r"$\rho'$ / mol/m$^3$")
    plt.tight_layout(pad=0.2)
    plt.savefig('water_nonmono.pdf')
    plt.close()

def profile_evaluation(FLD):
    expsL, expsV = [], []
    for jL in json.load(open(f'output/{FLD}_exps.json'))['jexpansions_rhoL']:
        eL = ChebTools.ChebyshevExpansion(jL['coef'], jL['xmin'], jL['xmax'])
        expsL.append(eL)
    for jV in json.load(open(f'output/{FLD}_exps.json'))['jexpansions_rhoV']:
        eV = ChebTools.ChebyshevExpansion(jV['coef'], jV['xmin'], jV['xmax'])
        expsV.append(eV)

    ceL = ChebTools.ChebyshevCollection(expsL)
    ceV = ChebTools.ChebyshevCollection(expsV)

    tic = timeit.default_timer()
    for i in range(50000):
        ceL(450.0)
    toc = timeit.default_timer()
    print((toc-tic)/50000)

def plot_widths(FLD):
    fig, ax = plt.subplots(1,1,figsize=(3.5, 3))
    for jL in json.load(open(f'output/{FLD}_exps.json'))['jexpansions_rhoL']:
        plt.plot(jL['xmin'], jL['xmax']-jL['xmin'], 'k.')
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$T_{\rm min}$ / K')
    plt.ylabel(r'$T_{\rm max}-T_{\rm min}$ / K')
    plt.xlim(250, 700)
    plt.gca().set_xticks([300, 400, 500, 600])
    plt.tight_layout(pad=0.2)
    plt.savefig('widths_of_intervals.pdf')
    plt.close()

def plot_worst():
    with PdfPages('devs.pdf') as PDF:
        with PdfPages('gooddevs.pdf') as goodPDF:
            for f in sorted(glob.glob('output/check/*.json')):
                
                fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(4,3))
            
                j = json.load(open(f))
                Tcrit = j['meta']['Tcrittrue / K']
                df = pandas.DataFrame(j['data'])
                df['errL'] = np.abs(df["rho'(SA) / mol/m^3"]/df["rho'(mp) / mol/m^3"]-1)
                df['errV'] = np.abs(df["rho''(SA) / mol/m^3"]/df["rho''(mp) / mol/m^3"]-1)
                df['Theta'] = (Tcrit-df['T / K'])/Tcrit

                # REFPROP calculations
                FLD = os.path.split(f)[1].split('.')[0].replace('_check',  '')
                
                if DEV and FLD not in ['R152A', 'PROPANE']: continue # for testing
                
                RP.SETFLUIDSdll(FLD)
                def add_REFPROP(row):
                    r = RP.REFPROPdll('','TQ','DLIQ;DVAP',RP.MOLAR_BASE_SI,0,0,row['T / K'],0,[1.0])
                    if r.ierr != 0:
                        return np.nan, np.nan
                    else:
                        return r.Output[0:2]
                df[['rhoL(RP) / mol/m^3', 'rhoV(RP) / mol/m^3']]  = df.apply(add_REFPROP, axis=1, result_type='expand')
                df['errLRP'] = np.abs(df["rhoL(RP) / mol/m^3"]/df["rho'(mp) / mol/m^3"]-1)
                df['errVRP'] = np.abs(df["rhoV(RP) / mol/m^3"]/df["rho''(mp) / mol/m^3"]-1)

                mask_normal = ((df['Theta'] > 1e-6) & (df["rho'(SA) / mol/m^3"]/df["rho''(SA) / mol/m^3"] < 1e14))
                df_lowT = df[mask_normal]
                df_crit = df[~mask_normal]
                good = False
                if np.max(df_lowT['errV']) < 1e-12 and np.max(df_lowT['errL']) < 1e-12:
                    good = True

                print(f, np.max(df_lowT['errL']), np.max(df_lowT['errV']), np.max(df_crit['errL']), np.max(df_crit['errV']))
                ax1.plot(1-df['T / K']/Tcrit, df['errL'], lw=0.2, label=r'$\Upsilon$=SA') 
                ax2.plot(1-df['T / K']/Tcrit, df['errV'], lw=0.2, label=r'$\Upsilon$=SA') 

                ax1.plot(1-df['T / K']/Tcrit, df['errLRP'], color='r', lw=0.2, label=r'$\Upsilon$=RP', dashes=[3,1,1,1]) 
                ax2.plot(1-df['T / K']/Tcrit, df['errVRP'], color='r', lw=0.2, label=r'$\Upsilon$=RP', dashes=[3,1,1,1])
                ax1.set_xscale('log')
                ax1.set_yscale('log')
                ax2.set_yscale('log')
                
                ax1.legend(loc='best')
                
                for ax in ax1, ax2:
                    ax.axhline(1e-12, dashes=[2,2], color='k', lw=0.5)
                    ax.axvline(1e-6, dashes=[2,2], color='k', lw=0.5)
                if ax1.get_xlim()[0] < 1e-10:
                    ax1.set_xlim(left=1e-10)
                ax1.set_ylabel(r"$|\rho'_{\Upsilon}/\rho'_{\rm ep}-1|$")
                ax2.set_ylabel(r"$|\rho''_{\Upsilon}/\rho''_{\rm ep}-1|$")
                ax2.set_xlabel(r'$\Theta\equiv (T_{\rm crit,num}-T)/T_{\rm crit,num}$')
                

                # if FLD in ['R152A', 'NF3']:
                #     df.to_csv(f'{FLD}_calcs.csv', index=False)

                plt.tight_layout(pad=0.2)
                xticks = ax.get_xticks()
                ax.set_xticks(xticks[0:len(xticks):2])
                for ax in ax1, ax2:
                    yticks = ax.get_yticks()
                    ax.set_yticks(yticks[0:len(yticks):2])
                
                ax1.set_ylim(1e-16, 1e-1)
                ax2.set_ylim(1e-16, 1e-1)
                ax1.set_xlim(right=1)
                for ax in ax1, ax2:
                    ax.text(1e-6/1.3, ax.get_ylim()[-1]/50, r'$10^{-6}$', ha='right', va='top')
                    ax.text(ax.get_xlim()[-1]/2, 1e-12, r'$10^{-12}$', ha='right', va='bottom')
                
                ax2.text(ax2.get_xlim()[-1], ax2.get_ylim()[-1], FLD, ha='right', va='top', bbox=dict(color='lightgrey', boxstyle='round,pad=0'))
                
                if good:
                    goodPDF.savefig(fig)
                else:
                    PDF.savefig(fig)
                plt.close()

def plot_pmu_devs():
    good_REFPROP, bad_REFPROP = 0, 0
    with PdfPages('pmu_devs.pdf') as PDF:
        for f in sorted(glob.glob('output/check/*.json')):
            
            fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(4,3))
        
            j = json.load(open(f))
            Tcrit = j['meta']['Tcrittrue / K']
            df = pandas.DataFrame(j['data'])
            df['errL'] = np.abs(df["rho'(SA) / mol/m^3"]/df["rho'(mp) / mol/m^3"]-1)
            df['errV'] = np.abs(df["rho''(SA) / mol/m^3"]/df["rho''(mp) / mol/m^3"]-1)
            df['Theta'] = (Tcrit-df['T / K'])/Tcrit

            FLD = os.path.split(f)[1].split('.')[0].replace('_check',  '')
            if DEV and FLD not in ['PROPANE','R13']: continue 
            
            model = teqp.build_multifluid_model([f'teqp_REFPROP10/dev/fluids/{FLD}.json'], teqp.get_datapath())
            z = np.array([1.0])
            R = model.get_R(z)
            
            def get_p(row, rhokey):
                T = row['T / K']
                rho = row[rhokey]
                return rho*R*T*(1+model.get_Ar01(T, rho, z))
            df['pLSA / Pa'] = df.apply(get_p, axis=1, rhokey="rho'(SA) / mol/m^3")
            df['pVSA / Pa'] = df.apply(get_p, axis=1, rhokey="rho''(SA) / mol/m^3")
            df['errP'] = np.abs(df["pLSA / Pa"]/df["pVSA / Pa"]-1)

            def add_pdev_REFPROP(row):
                r = RP.REFPROPdll(FLD,'TQ','DLIQ;DVAP;PLIQ;PVAP;T',RP.MOLAR_BASE_SI,0,0,row['T / K'],0,[1.0])
                if r.ierr == 0:
                    pl, pv = r.Output[2:4]
                    # print(pl/pv-1)
                    return np.abs(pl/pv-1)
                else:
                    print(r.herr)
                    return np.nan
            df['errP(REFPROP)'] = df.apply(add_pdev_REFPROP, axis=1)

            def get_VLEmu(row, rhokey):
                T = row['T / K']
                rho = row[rhokey]
                return model.get_Ar00(T, rho, z) + model.get_Ar01(T, rho, z)
            df['muVLELSA'] = df.apply(get_VLEmu, axis=1, rhokey="rho'(SA) / mol/m^3")
            df['muVLEVSA'] = df.apply(get_VLEmu, axis=1, rhokey="rho''(SA) / mol/m^3")
            df['errmu'] = np.abs(df["muVLELSA"]-df["muVLEVSA"]+np.log(df["rho'(SA) / mol/m^3"]/df["rho''(SA) / mol/m^3"]))

            def add_mudev_REFPROP(row):
                r = RP.REFPROPdll(FLD,'TQ','DLIQ;DVAP;GLIQ;GVAP;T;R',RP.MOLAR_BASE_SI,0,0,row['T / K'],0,[1.0])
                if r.ierr == 0:
                    mul, muv, R, T = r.Output[2:6]
                    # print(pl/pv-1)
                    return np.abs((mul-muv)/(R*T))
                else:
                    print(r.herr)
                    return np.nan
            df['errmu(REFPROP)'] = df.apply(add_mudev_REFPROP, axis=1)

            ax1.plot(1-df['T / K']/Tcrit, df['errP'], lw=0.2, label='SA') 
            ax1.plot(1-df['T / K']/Tcrit, df['errP(REFPROP)'], lw=0.2, color='r', label='REFPROP', dashes=[3,1,1,1]) 
            ax2.plot(1-df['T / K']/Tcrit, df['errmu'], lw=0.2)
            ax2.plot(1-df['T / K']/Tcrit, df['errmu(REFPROP)'], lw=0.2, color='r', dashes=[3,1,1,1]) 

            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax2.set_yscale('log')
            ax1.legend(loc='best')
            ax1.set_ylim(1e-17, 100)
            ax2.set_ylim(1e-17, 100)
            
            # plt.suptitle(FLD)
            for ax in ax1, ax2:
                ax.axhline(1e-12, dashes=[2,2], color='k', lw=0.5)
                ax.axvline(1e-6, dashes=[2,2], color='k', lw=0.5)
            if ax1.get_xlim()[0] < 1e-10:
                ax1.set_xlim(left=1e-10)
            ax1.set_ylabel(r"$r_p$")
            ax2.set_ylabel(r"$r_\mu$")
            ax2.set_xlabel(r'$\Theta\equiv (T_{\rm crit,num}-T)/T_{\rm crit,num}$')
            
            xticks = ax.get_xticks()
            ax.set_xticks(xticks[0:len(xticks):2])
            for ax in ax1, ax2:
                yticks = ax.get_yticks()
                ax.set_yticks(yticks[0:len(yticks):2])
            ax1.set_ylim(1e-16, 1)
            ax2.set_ylim(1e-16, 1e-8)
            ax2.text(ax2.get_xlim()[-1], ax2.get_ylim()[-1], FLD, ha='right', va='top', bbox=dict(color='lightgrey', boxstyle='round,pad=0'))

            plt.tight_layout(pad=0.2)
            
            for ax in ax1, ax2:
                ax.text(1e-6/1.3, ax.get_ylim()[-1]/50, r'$10^{-6}$', ha='right', va='top')
                ax.text(1e-4, 1e-12, r'$10^{-12}$', ha='right', va='bottom')
                    
            PDF.savefig(fig)
            plt.close()

            good_REFPROP += sum(np.isfinite(df['errP(REFPROP)']))
            bad_REFPROP += sum(~np.isfinite(df['errP(REFPROP)']))

    print(abs(bad_REFPROP/(bad_REFPROP+good_REFPROP)-1), '% of calculations fail', bad_REFPROP, good_REFPROP)

def test_inverse_functions():
    for f in sorted(glob.glob('output/check/*.json')):
        FLD = os.path.split(f)[1].split('.')[0].replace('_check',  '')
        # if FLD != 'MXYLENE': continue

        j = json.load(open(f))
        Tcrit = j['meta']['Tcrittrue / K']
        rhocrit = j['meta']['rhocrittrue / mol/m^3']
        df = pandas.DataFrame(j['data'])
        df['Theta'] = (Tcrit-df['T / K'])/Tcrit

        # model = teqp.build_multifluid_model([FLD], 'teqp_REFPROP10')
        # Tcnum, rhocnum = model.solve_pure_critical(j['meta']['Tcrit / K'], j['meta']['rhocrittrue / mol/m^3']*1.01)
        # if abs(Tcnum/Tcrit-1) > 1e-10:
        #     print(FLD, Tcnum, Tcrit)
        # if abs(rhocnum/rhocrit-1) > 1e-10:
        #     print(FLD, rhocnum, rhocrit)

        ceL, ceV, cep = get_expansions(FLD, and_p=True)

        for ce in cep.get_exps():
            xx = np.linspace(ce.xmin(), ce.xmax(), 1000000)
            yy = ce.y(xx)
            all_increasing = all(np.diff(yy) > 0)
            if not all_increasing:
                print('[P]:', FLD, ce.xmin(), ce.xmax(), ce.deriv(1).has_real_roots_Descartes(1e-10), ce.is_monotonic(), all_increasing)
                plt.plot(xx, yy)
                plt.savefig(f'nonmono_p_{FLD}_{ce.xmin()}.pdf')
                plt.close()

        # Check that all T(p) calcs work with superancillary
        xmin = cep.get_exps()[0].xmin()
        xmax = cep.get_exps()[-1].xmax()
        xx = np.linspace(xmin, xmax, 100000)
        yy = [cep(x_) for x_ in xx]
        func_calls = []
        for i, y in enumerate(yy):
            resid = lambda T: (cep(T)-y)/y
            try:
                TBrent, info = scipy.optimize.brentq(resid, xmin, xmax, full_output=True)
                if abs(TBrent/xx[i]-1) > 1e-10:
                    raise ValueError(y)
                else:
                    func_calls.append(info.function_calls)
            except BaseException as be:
                print(cep(xmin) - y)
                print(cep(xmax) - y)
                print('[BRENTP]:', FLD, y, be)
        print(FLD, np.mean(func_calls), 'function calls needed on average')

        continue
        
        for ce in ceV.get_exps():
            TT = np.linspace(ce.xmin(), ce.xmax(), 10000)
            yy = ce.y(TT)
            Theta = (Tcrit-TT)/Tcrit
            all_increasing = all(np.diff(yy) > 0)
            if not all_increasing:
                print('[V]:', FLD, ce.xmin(), ce.xmax(), ce.deriv(1).has_real_roots_Descartes(1e-10), ce.is_monotonic(), all_increasing)
                print(j['meta'] )
                plt.plot(df['Theta'], df["rho''(mp) / mol/m^3"], '.')
                plt.plot(Theta, yy)
                for iarg, d in enumerate(np.diff(yy)):
                    if d <= 0:
                        plt.axvline(Theta[iarg])
                        print(Theta[iarg])
                if np.min(Theta) < 0.1:
                    plt.xscale('log')
                plt.xlim(np.min(Theta), np.max(Theta))
                plt.ylim(np.min(yy), np.max(yy))
                
                # plt.axvline(Tcrit, dashes=[2,2])
                plt.title('rhoV')
                plt.savefig(f'nonmono_rhoV_{FLD}_{ce.xmin()}.pdf')
                plt.close()

        for ce in ceL.get_exps():
            TT = np.linspace(ce.xmin(), ce.xmax(), 10000)
            yy = ce.y(TT)
            Theta = (Tcrit-TT)/Tcrit
            all_decreasing = all(np.diff(yy) < 0)
            if not all_decreasing:
                print('[L]:', FLD, ce.xmin(), ce.xmax(), ce.deriv(1).has_real_roots_Descartes(1e-10), ce.is_monotonic(), all_decreasing)
                plt.plot(df['Theta'], df["rho'(mp) / mol/m^3"], '.')
                plt.plot(Theta, yy)
                for iarg, d in enumerate(np.diff(yy)):
                    if d >= 0:
                        plt.axvline(Theta[iarg])
                        print(Theta[iarg])

                if np.min(Theta) < 0.1:
                    plt.xscale('log')
                plt.xlim(np.min(Theta), np.max(Theta))
                plt.ylim(np.min(yy), np.max(yy))
                # plt.axvline(Tcrit, dashes=[2,2])
                plt.title('rhoL')
                plt.savefig(f'nonmono_rhoL_{FLD}_{ce.xmin()}.pdf')
                plt.close()

def whyrt4():

    FLD = 'PROPANE'
    f = f'output/check/{FLD}_check.json'

    j = json.load(open(f))
    Tcrit = j['meta']['Tcrittrue / K']
    df = pandas.DataFrame(j['data'])
    df['Theta'] = (Tcrit-df['T / K'])/Tcrit

    _, __, cep = get_expansions(FLD, and_p=True)
    p = [cep(T) for T in df['T / K']]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(df['T / K'], p)
    ax2.plot(df['T / K'], np.log10(p))
    ax3.plot(df['T / K'], np.array(p)**0.25)
    print('min,max of p: ', [f(np.array(p)) for f in (min, max)])
    print('min,max of p^{1/4}: ', [f(np.array(p)**0.25) for f in (min, max)])
    ax3.set_xlabel('$T$ / K')
    ax1.set_ylabel(r'$p$ / Pa')
    ax2.set_ylabel(r'$\log_{10}(p~/~{\rm Pa})$')
    ax3.set_ylabel(r'$(p~/~{\rm Pa})^{1/4}$')
    plt.tight_layout(pad=0.2)
    plt.savefig('whyrt4_PROPANE.pdf')
    plt.close()

    N = 120
    j = np.arange(0, N+1)
    x = np.cos(j*np.pi/N)
    print(x)
    pmin = p[0]
    pmax = p[-1]
    pnodes = (pmax-pmin)/2*x + (pmax+pmin)/2
    print(pnodes, pmin, pmax, pnodes[0]/pmax, pnodes[-1]/pmin)

def plot_invpanc_devs():
    with PdfPages('invpanc_devs.pdf') as PDF:
        for f in sorted(glob.glob('output/check/*.json')):

            fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6,3), width_ratios=[2,2])
            
            FLD = os.path.split(f)[1].split('.')[0].replace('_check',  '')
            # if FLD != 'PROPANE': continue
            print('inv(p) for ' + FLD)
            if DEV and FLD not in ['R152A', 'PROPANE']: continue # for testing

            j = json.load(open(f))
            Tcrit = j['meta']['Tcrittrue / K']
            df = pandas.DataFrame(j['data'])
            df['Theta'] = (Tcrit-df['T / K'])/Tcrit

            ceL, ceV, cep = get_expansions(FLD, and_p=True)
            Tmin = cep.get_exps()[0].xmin()
            Tmax = cep.get_exps()[-1].xmax()
            pmin = cep(Tmin)
            pmax = cep(Tmax)

            n = 4.0
            def get_T(rt4p):
                if rt4p <= pmin**(1/n): return Tmin
                if rt4p >= pmax**(1/n): return Tmax
                p = rt4p**n
                def objective(T):
                    return (cep(T)-p)/p
                fL = objective(Tmin)
                fR = objective(Tmax)
                if abs(fL) < 1e-10:
                    return Tmin
                if abs(fR) < 1e-10:
                    return Tmax
                return scipy.optimize.brentq(objective, Tmin, Tmax)
            def callback(i, exps):
                # print(i)
                pass
            invrt4pBrent = ChebTools.ChebyshevCollection(ChebTools.dyadic_splitting(12, get_T, pmin**(1/n), pmax**(1/n), 3, 1e-13, 20, callback))
                    
            # def add_Troundtrip(row):
            #     try:
            #         # Evaluate the expansion
            #         p = cep(row['T / K'])
            #         # Evaluate the inverse function to get T
            #         return invp.y_unsafe(p)
            #     except BaseException as be:
            #         print('[Ttroundtrip]:', be)
            #         return np.nan
            # df['Troundtrip(SA/inv) / K'] = df.apply(add_Troundtrip, axis=1)
            
            def add_Troundtrip(row):
                try:
                    # Evaluate the expansion
                    p = cep(row['T / K'])
                    # Evaluate the inverse function to get T
                    # with p^{1/4} as independent variable
                    return invrt4pBrent.y_unsafe(p**(1/n))
                except BaseException as be:
                    print('[Ttroundtrip/SA/invBrentrt4]:', be, '@', row.to_dict())
                    return np.nan
            df['Troundtrip(SA/invBrentrt4) / K'] = df.apply(add_Troundtrip, axis=1)
            
            def add_TroundtripBrent(row):
                try:
                    # Evaluate the expansion
                    p = cep(row['T / K'])
                    # Use Brent's method on expansion itself to get the value
                    return scipy.optimize.brentq(lambda T: (cep(T)-p)/p, Tmin, Tmax)
                except BaseException as be:
                    print('[Ttroundtrip/Brent]:', be, '@', row.to_dict())
                    return np.nan
            df['Troundtrip(SA/Brent) / K'] = df.apply(add_TroundtripBrent, axis=1)

            RP.SETFLUIDSdll(FLD)
            RP.FLAGSdll('R', 2)
            def add_Troundtrip_REFPROP(row):
                r = RP.REFPROPdll('','TQ','P',RP.MOLAR_BASE_SI,0,0,row['T / K'],0,[1.0])
                p = r.Output[0]
                if r.ierr != 0:
                    return np.nan
                r = RP.REFPROPdll('','PQ','T',RP.MOLAR_BASE_SI,0,0,p,0,[1.0])
                if r.ierr != 0:
                    return np.nan
                return r.Output[0]
            df['Troundtrip(REFPROP) / K'] = df.apply(add_Troundtrip_REFPROP, axis=1)

            for ax in axes:
                # ax.plot(df['Theta'], np.abs(df['Troundtrip(SA/inv) / K'] - df['T / K']), label='SA/inv')
                ax.plot(df['Theta'], np.abs(df['Troundtrip(SA/invBrentrt4) / K'] - df['T / K']), label=r'SA/invBrent($\sqrt[4]{p}$)')
                ax.plot(df['Theta'], np.abs(df['Troundtrip(SA/Brent) / K'] - df['T / K']), label='SA/Brent')
                ax.plot(df['Theta'], np.abs(df['Troundtrip(REFPROP) / K'] - df['T / K']), label='REFPROP')

            axes[0].set_xscale('log')
            axes[1].set_xscale('linear')

            # plt.suptitle(FLD)
            axes[0].set_ylabel(r'$|T_{\rm roundtrip} - T_{\rm orig}|$ / K')
            axes[0].set_ylim(1e-17, 100)
            axes[0].set_yscale('log')
            plt.tight_layout(pad=0.2)
            axes[0].legend(loc='best')
            Thetasplit = 0.1
            axes[0].set_xlim(right=Thetasplit)
            axes[1].set_xlim(left=Thetasplit)
            xticks = axes[0].get_xticks()
            axes[0].set_xticks([x for x in xticks[0:len(xticks):2] if x <= Thetasplit])

            plt.tight_layout(pad=0.2,rect=[0.0, 0.07, 1, 0.99])
            ax2 = axes[1]
            ax2.text(ax2.get_xlim()[-1], ax2.get_ylim()[-1], FLD, ha='right', va='top', bbox=dict(color='lightgrey', boxstyle='round,pad=0'))

            # add a big axis, hide frame
            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axis
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel(r'$\Theta\equiv (T_{\rm crit,num}-T)/T_{\rm crit,num}$')

            PDF.savefig(plt.gcf())
            plt.close()

def plot_panc_devs():
    with PdfPages('panc_from_rhoanc_devs.pdf') as PDF:
        for f in sorted(glob.glob('output/check/*.json')):
            
            FLD = os.path.split(f)[1].split('.')[0].replace('_check',  '')
            # if FLD != 'WATER': continue 
            # if FLD not in ['PROPANE','R13']: continue
            if DEV and FLD not in ['R152A', 'PROPANE']: continue # for testing
            ceL, ceV = get_expansions(FLD, and_p=False)

            fig, axes = plt.subplots(1,2,sharey=True,figsize=(6,3), width_ratios=[2,2])
        
            j = json.load(open(f))
            Tcrit = j['meta']['Tcrittrue / K']
            REOS = j['meta']['gas_constant / J/mol/K']
            df = pandas.DataFrame(j['data'])
            df['Theta'] = (Tcrit-df['T / K'])/Tcrit
            Tmin = ceL.get_exps()[0].xmin()

            # Calculations in extended precision
            df['(p/R)_ep'] = df['p(mp) / Pa']/REOS

            model = teqp.build_multifluid_model([f'teqp_REFPROP10/dev/fluids/{FLD}.json'], teqp.get_datapath())
            z = np.array([1.0])
            R = model.get_R(z)
            print(R, FLD, REOS)
            if (abs(R/REOS-1) > 1e-12):
                raise ValueError('R does not match perfectly')
            print('{0:20.16f} {1:20.16f}'.format(ceL.get_exps()[0].xmin(), ceL.get_exps()[-1].xmax()))
            print('{0:20.16f} {1:20.16f}'.format(Tmin+1e-10, Tcrit-1e-10))
            print('{0:20.16f} {1:20.16f}'.format(df['T / K'].iloc[0], df['T / K'].iloc[-1]))
            
            # def get_p(row, rhokey):
            #     T = row['T / K']
            #     rho = row[rhokey]
            #     return rho*T*(1+model.get_Ar01(T, rho, z))
            # df['pLSA/R'] = df.apply(get_p, axis=1, rhokey="rho'(SA) / mol/m^3")
            # df['pVSA / Pa'] = df.apply(get_p, axis=1, rhokey="rho''(SA) / mol/m^3")            

            def build_panc_from_rhoanc(W):
                """ W: weighting parameter of the pressures"""
                def get_p(T):
                    rhoL = ceL(T)
                    rhoV = ceV(T)
                    pL = rhoL*R*T*(1+model.get_Ar01(T, rhoL, z))
                    pV = rhoV*R*T*(1+model.get_Ar01(T, rhoV, z))
                    if W == 0:
                        return pL 
                    elif W == 1:
                        return pV 
                    else:
                        return W*pV + (1-W)*pL
                return ChebTools.ChebyshevCollection(ChebTools.dyadic_splitting(12, get_p, Tmin, Tcrit, 3, 1e-12, 12))
            panc = build_panc_from_rhoanc(W=0.5)
            pancL = build_panc_from_rhoanc(W=0)
            pancV = build_panc_from_rhoanc(W=1)

            def evalp(row, anc):
                try: 
                    return anc(row['T / K']) 
                except BaseException as be:
                    print(be) 
                    return np.nan
            df['panc(T)/R'] = df.apply(evalp, axis=1, anc=panc)/R
            df['pancL(T)/R'] = df.apply(evalp, axis=1, anc=pancL)/R
            df['pancV(T)/R'] = df.apply(evalp, axis=1, anc=pancV)/R
            for ax in axes:
                # ax.plot(df['Theta'], np.abs(df['panc(T)/R']/df['(p/R)_ep']-1), color='b', label=r"$[\frac{1}{2}p(\rho'_{\rm SA}(T))+\frac{1}{2}p(\rho''_{\rm SA}(T))]_{\rm SA}$", dashes=[2,2])
                ax.plot(df['Theta'], np.abs(df['pancL(T)/R']/df['(p/R)_ep']-1), color='c', label=r"$p_{\rm SA}(\rho'_{\rm SA}(T))$", dashes=[2,2])
                ax.plot(df['Theta'], np.abs(df['pancV(T)/R']/df['(p/R)_ep']-1), color='orange', label=r"$p_{\rm SA}(\rho''_{\rm SA}(T))$", dashes=[2,2])
                ax.plot(df['Theta'], np.abs((df['p(SA) / Pa']/REOS)/df['(p/R)_ep']-1), color='green', label=r'$p_{\rm SA}(T)$')

            RP.SETFLUIDSdll(FLD)
            RP.FLAGSdll('R', 2)
            errmsgs = []
            def add_poverR_REFPROP(row):
                r = RP.REFPROPdll('','TQ','DLIQ;DVAP;PLIQ;PVAP;T;R',RP.MOLAR_BASE_SI,0,0,row['T / K'],0,[1.0])
                if r.ierr == 0:
                    pl, pv = r.Output[2:4]
                    R = r.Output[5]
                    return pl/R
                else:
                    # print(r.herr)
                    errmsgs.append(r.herr.strip())
                    return np.nan
            for msg in set(errmsgs):
                print(msg)
            df['p(REFPROP)/R'] = df.apply(add_poverR_REFPROP, axis=1)
            for ax in axes:
                ax.plot(df['Theta'], np.abs(df['p(REFPROP)/R']/df['(p/R)_ep']-1), color='r', label='REFPROP', dashes=[3,3])

            ax1, ax2 = axes

            ax1.set_xscale('log')
            ax1.set_yscale('log')
            # ax2.set_yscale('log')
            ax1.legend(loc='best')

            Thetasplit = 0.1
            if ax1.get_xlim()[0] < 1e-10:
                ax1.set_xlim(left=1e-10)
            ax1.set_xlim(ax1.get_xlim()[0], Thetasplit)
            ax2.set_xlim(Thetasplit, ax2.get_xlim()[1])
            
            # plt.suptitle(FLD)            
            ax1.set_ylabel(r"$|p_{\rm calc}/p_{\rm ep}-1|$")
            # ax1.set_xlabel(r'$\Theta\equiv (T_{\rm crit,num}-T)/T_{\rm crit,num}$')

            for ax in [ax1, ax2]:
                xticks = ax.get_xticks()
                ax.set_xticks(xticks[0:len(xticks):2])
            for ax in [ax1]:
                ax.set_yticks(10.0**np.arange(-17, 2, 2))
            ax1.set_xlim(ax1.get_xlim()[0], Thetasplit)
            ax1.set_ylim(1e-17, 100)
            xticks = [t for t in ax1.get_xticks() if t <= Thetasplit-1e-6]
            ax1.set_xticks(xticks[0:len(xticks)])

            plt.tight_layout(pad=0.2,rect=[0.0,0.07,1,0.99])
            ax2.text(ax2.get_xlim()[-1], ax2.get_ylim()[-1], FLD, ha='right', va='top', bbox=dict(color='lightgrey', boxstyle='round,pad=0'))
            # fig.text(1, 0.5, rotation=90, s=FLD,ha='right', va='center')

            # add a big axis, hide frame
            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axis
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel(r'$\Theta\equiv (T_{\rm crit,num}-T)/T_{\rm crit,num}$')
            # plt.title(FLD)
            
            PDF.savefig(fig)
            plt.close()

            # break

def map_pages(PDFs):
    """ Find all the strings in sets of PDF and cache the locations in the files """
    maps = {}
    for PDF in PDFs:
        reader = PdfReader(PDF)
        outputs = []
        for ipage, page in enumerate(reader.pages):
            # page = reader.pages[0]
            for text_item in page.extract_text().splitlines():
                try:
                    float(text_item)
                except:
                    not_FLD = ['r','rp','=SA','=RP','(Tcrit,num T)/Tcrit,num']
                    if text_item in not_FLD: continue
                    print(text_item, ipage+1)
                    outputs.append({'string': text_item, 'pagenum': ipage+1})
        maps[PDF] = outputs 
    return maps

def make_SI_figs(*, mapcache):

    def get_devPDF(FLD, candidates=['gooddevs.pdf','devs.pdf']):
        for PDF, mapping in json.load(open(mapcache)).items():
            if PDF not in candidates: continue
            for entry in mapping:
                if FLD == entry['string']:
                    return PDF, entry['pagenum']
        raise KeyError(FLD)
                
    output = ''
    for i, f in enumerate(sorted(glob.glob('output/check/*.json'))):
        FLD = os.path.split(f)[1].split('.')[0].replace('_check',  '')
        # Lookup the name of the deviation file
        PDFrhodev, pagenumdev = get_devPDF(FLD)
        _, pagepmu = get_devPDF(FLD, candidates='pmu_devs.pdf')

        header = r"""
        \begin{figure}[H] %INS% 
        \centering
        """
        footer = r"""\caption{%caption%}
    \end{figure}""".replace("%caption%", FLD)
        
        # o = header.replace("%INS%", "\ContinuedFloat" if i != 0 else "")
        o = header.replace("%INS%", "")
        o += r"""\subcaptionbox{Orthobaric density deviations}{
            \includegraphics[width=2.5in,page=%pagenumdev%]{%PDFrhodev%}
        }\subcaptionbox{Phase equilibrium conditions}{
            \includegraphics[width=2.5in,page=%pagepmu%]{pmu_devs}
        }
        """.replace("%FLD%", FLD).replace('%root%',root).replace('%pagenumdev%', str(pagenumdev)).replace('%PDFrhodev%', PDFrhodev).replace('%pagepmu%',str(pagepmu)) + '\n'
        o += footer
        output += o
    return output

def make_SI_figs_p(*, mapcache):

    def get_devPDF(FLD, candidates=['panc_from_rhoanc_devs.pdf','invpanc_devs.pdf']):
        for PDF, mapping in json.load(open(mapcache)).items():
            if PDF not in candidates: continue
            for entry in mapping:
                if FLD == entry['string']:
                    return PDF, entry['pagenum']
        raise KeyError(FLD)
                
    output = ''
    for i, f in enumerate(sorted(glob.glob('output/check/*.json'))):
        FLD = os.path.split(f)[1].split('.')[0].replace('_check',  '')
        pagenum = i+1

        header = r"""
        \begin{figure}[H] %INS% 
        \centering
        """
        footer = r"""\caption{%caption%}
    \end{figure}""".replace("%caption%", FLD)
        
        # o = header.replace("%INS%", "\ContinuedFloat" if i != 0 else "")
        o = header.replace("%INS%", "")
        o += r"""\subcaptionbox{Pressure deviations}{
            \includegraphics[width=2.5in,page=%pagenumdev%]{panc_from_rhoanc_devs}
        }\subcaptionbox{Inverse pressure roundtrip error}{
            \includegraphics[width=2.5in,page=%pagepmu%]{invpanc_devs}
        }
        """.replace("%FLD%", FLD).replace('%root%',root).replace('%pagenumdev%', str(pagenum)).replace('%pagepmu%',str(pagenum)) + '\n'
        o += footer
        output += o
    return output

def plot_ancillary(FLD):
    
    class CritAncillary:
        def __init__(self, *, FLD):
            j = json.load(open(f'output/{FLD}_exps.json'))
            self.Tcritnum = j['meta']["Tcrittrue / K"]
            self.rhocritnum = j['meta']["rhocrittrue / mol/m^3"]
            self.cL = j['crit_anc']['cL']
            self.cV = j['crit_anc']['cV']

        def __call__(self, T):
            Theta = (self.Tcritnum-T)/self.Tcritnum
            rhoL = self.rhocritnum + np.exp(sum(c_i*np.log(Theta)**i for i, c_i in enumerate(self.cL)))
            rhoV = self.rhocritnum - np.exp(sum(c_i*np.log(Theta)**i for i, c_i in enumerate(self.cV)))
            return rhoL, rhoV
        
    class ConventionalAncillary:
        def __init__(self, *, FLD):
            model = teqp.build_multifluid_model([f'teqp_REFPROP10/dev/fluids/{FLD}.json'], teqp.get_datapath())
            self.anc = model.build_ancillaries()

        def __call__(self, T):
            try:
                return self.anc.rhoL(T), self.anc.rhoV(T)
            except:
                return np.nan, np.nan
        
    critanc = CritAncillary(FLD=FLD)
    convanc = ConventionalAncillary(FLD=FLD)

    # # Compare the ancillaries against each other
    # o = []
    # for Theta in np.geomspace(0.01, 1e-10, 1000):
    #     T = critanc.Tcritnum*(1-Theta)
    #     rhoLcrit, rhoVcrit = critanc(T)
    #     rhoLconv, rhoVconv = convanc(T)
    #     o.append({
    #         'Theta': Theta,
    #         'rhoLcrit': rhoLcrit,
    #         'rhoVcrit': rhoVcrit,
    #         'rhoLconv': rhoLconv,
    #         'rhoVconv': rhoVconv,
    #     })
    # df = pandas.DataFrame(o)
    # plt.plot(df['Theta'], np.abs(df['rhoLcrit']/df['rhoLconv']-1), label='liquid')
    # plt.plot(df['Theta'], np.abs(df['rhoVcrit']/df['rhoVconv']-1), label='vapor', dashes=[2,2])
    # plt.xscale('log')
    # plt.yscale('log')

    # plt.gca().set(xlabel=r'$\Theta = (T_{\rm crit, num}-T)/T_{\rm crit, num}$', ylabel=r'$|\rho^{\pi}_{\rm crit}/\rho^{\pi}_{\rm conv}-1|$')
    # plt.legend()
    # plt.tight_layout(pad=0.2)
    # plt.savefig('ancillary_crosscomparison.pdf')
    # plt.close()

    # Compare the ancillaries against extended precision calculations
    o = []
    for row in json.load(open(f'output/check/{FLD}_check.json'))['data']:
        T = row['T / K']
        Theta = (critanc.Tcritnum-T)/critanc.Tcritnum
        if Theta > 0.01: continue
        rhoLcrit, rhoVcrit = critanc(T)
        rhoLconv, rhoVconv = convanc(T)
        try:
            rhoLmp = row["rho'(mp) / mol/m^3"]
            rhoVmp = row["rho''(mp) / mol/m^3"]
            o.append({
                'Theta': Theta,
                'rhoLcrit': rhoLcrit,
                'rhoVcrit': rhoVcrit,
                'rhoLconv': rhoLconv,
                'rhoVconv': rhoVconv,
                'rhoLmp': rhoLmp,
                'rhoVmp': rhoVmp,
            })
        except:
            pass
    plt.subplots(figsize=(3.5, 3.5))
    df = pandas.DataFrame(o)
    plt.plot(df['Theta'], np.abs(df['rhoLcrit']/df['rhoLmp']-1), 'r-', label='liquid(crit)', lw=3)
    plt.plot(df['Theta'], np.abs(df['rhoVcrit']/df['rhoVmp']-1), 'b-', label='vapor(crit)', dashes=[2,2], lw=3)

    plt.plot(df['Theta'], np.abs(df['rhoLconv']/df['rhoLmp']-1), 'r--', label='liquid(conv)')
    plt.plot(df['Theta'], np.abs(df['rhoVconv']/df['rhoVmp']-1), 'b--', label='vapor(conv)', dashes=[2,2])
    plt.xscale('log')
    plt.yscale('log')

    plt.gca().set(xlabel=r'$\Theta = (T_{\rm crit, num}-T)/T_{\rm crit, num}$', ylabel=r'$|\rho^{\pi}_{\rm anc}/\rho^{\pi}_{\rm ep}-1|$')
    plt.legend(loc='best')
    plt.tight_layout(pad=0.2)
    xticks = plt.gca().get_xticks()
    plt.gca().set_xticks(xticks[0:len(xticks):2])
    plt.ylim(1e-6, 1e-1)
    plt.savefig('ancillary_boost.pdf')
    plt.show()

def check_all_conversions(RP, FLUIDS):

    for FLD in glob.glob(FLUIDS+'/*.FLD'):
        RP.FLAGSdll('R', 2)
        RP.SETFLUIDSdll(FLD)
        RP.FLAGSdll('R', 2)
        FLD = os.path.split(FLD)[1].split('.')[0]

        path = 'teqp_REFPROP10/dev/fluids/'+FLD+'.json'
        if not os.path.exists(path):
            print(path)
        model = teqp.build_multifluid_model([path], teqp.get_datapath())

        Tcrit, rhocrit = RP.REFPROPdll('','','TCRIT;DCRIT',RP.MOLAR_BASE_SI,0,0,0,0,[1.0]).Output[0:2]

        T = Tcrit*1.1
        z = np.array([1.0])

        for rho in np.geomspace(1e-10, 1.5*rhocrit):
            r = RP.REFPROPdll('','TD&','P;R',RP.MOLAR_BASE_SI,0,0,T,rho,[1.0])
            pR_REFPROP = r.Output[0]/r.Output[1]
            pR_teqp = rho*T*(1+model.get_Ar01(T,rho,z))
            err = 100*(pR_teqp/pR_REFPROP-1)
            if abs(err) > 1e-12:
                print(FLD, err, pR_REFPROP, pR_teqp, T, rho)

            if FLD == 'NF3':
                o = RP.REFPROPdll('/Users/ihb/Desktop/NF3BWRfixed.FLD','TD&','P;R',RP.MOLAR_BASE_SI,0,0,T,rho,[1.0])
                print(pR_REFPROP, pR_teqp, o.Output[0]/o.Output[1]) 
                o = RP.REFPROPdll('/Users/ihb/Desktop/NF3fromsandbox.FLD','TD&','P;R',RP.MOLAR_BASE_SI,0,0,T,rho,[1.0])
                print(pR_REFPROP, pR_teqp, o.Output[0]/o.Output[1])
                o = RP.REFPROPdll('/Users/ihb/Desktop/NF3fromfixedversion10.0.FLD','TD&','P;R',RP.MOLAR_BASE_SI,0,0,T,rho,[1.0])
                print(pR_REFPROP, pR_teqp, o.Output[0]/o.Output[1])

def make_fluid_info_table(ref):
    dfsugg = pandas.read_csv('SuggestedNames.csv', index_col='FLD')
    
    o = []
    for FLD in ref.FLDs:
        key = ref.keymap.get(FLD, None)
        if key is None:
            refstring = ''
        else:
            refstring = r'Ref. \citenum{' + key + '}' 
            if key in ref.newdois:
                refstring += '(n)'
        suggested = dfsugg.loc[FLD, 'SuggestedName']
        
        o.append({
            'REFPROP name': FLD,
            'name': suggested,
            'ref': refstring
        })
    caption = r"""
    Equations of state considered in this work. \listsumdelim All EOS coefficients 
    are taken from REFPROP 10.0. The reference information was looked up by digital object identifier(doi) when available, or looked up from CoolProp \cite{Bell-IECR-2014} for 
    books without a doi. EOS published in the literature after the release of REFPROP 10.0 are indicated by (n), and the coefficients 
    are assumed to be the same as in REFPROP 10.0.
    """
    caption = ' '.join(caption.split('\n'))
    df = pandas.DataFrame(o)
    with pandas.option_context("max_colwidth", 1000):
        with open('EOS_info.tex.in', 'w') as fp:
            contents = df.to_latex(index=False, caption=caption, longtable=True, label='tab:EOSlist', escape=False)
            contents = contents.replace('begin{longtable}', 'begin{longtable*}').replace('end{longtable}', 'end{longtable*}')
            fp.write(contents)
            
    o = []
    for FLD in ref.FLDs:
        key = ref.keymap.get(FLD, None)
        if key is None:
            refstring = ''
        else:
            refstring = r'Ref. \citenum{' + key + '}' 
            if key in ref.newdois:
                refstring += '(n)'

        ceL, ceV, cep = get_expansions(FLD=FLD, and_p=True)
        Tcheck = np.floor(cep.get_exps()[-1].xmax()*0.9)
        check_vals = f'{ceL(Tcheck):20.12e}',f'{ceV(Tcheck):20.12e}',f'{cep(Tcheck):20.12e}'
        check_val_keys = "$\rho'$ / mol/m$^3$","$\rho''$ / mol/m$^3$","$p$ / Pa"
        
        o.append({
            'REFPROP name': FLD,
            '$T$ / K': Tcheck,
        } 
        | 
        {k:v for k,v in zip(check_val_keys, check_vals)}
        )
    caption = r"""
    Check values calculated from the superancillary functions. The temperature considered is nominally $0.9T_{\rm crit}$, rounded down to the next integer. 
    """
    caption = ' '.join(caption.split('\n'))
    df = pandas.DataFrame(o)
    with pandas.option_context("max_colwidth", 1000):
        with open('check_values.tex.in', 'w') as fp:
            fp.write(df.to_latex(index=False, caption=caption, longtable=True, label='tab:checkvalues', escape=False))

def cleanupbibtex(BibTeXfile):
    f = open(BibTeXfile, encoding='utf-8').read()
    journal_map = {
        'Journal of Physical and Chemical Reference Data': 'J. Phys. Chem. Ref. Data',
        'Journal of Chemical and Engineering Data': 'J. Chem. Eng. Data',
        'Fluid Phase Equilibria': 'Fluid Phase Equilib.',
        'Thermal Engineering': 'Therm. Eng.',
        'International Journal of Thermophysics': 'Int. J. Thermophys.',
        'Chemical Engineering Science': 'Chem. Eng. Sci.',
        'Industrial and Engineering Chemistry Research': 'Ind. Eng. Chem. Res.'
    }
    for k,v in journal_map.items():
        f = f.replace(k,v)

    # Protect title capitalization with outer { }
    import re
    f = re.sub(r'title\s+= \{(.+)\}', r'title = {{\1}}', f)
    
    # Fix typos
    for old, new in {'625K': '625 K', '150MPa': '150 MPa', 'trans-1':'$trans$-1', 'Cv,': 'C$_v$,'}.items():
        f = f.replace(old, new)

    with open(BibTeXfile, 'w', encoding='utf-8') as fp:
        fp.write(f)

if __name__ == '__main__':

    root = os.getenv('RPPREFIX')
    if root == '':
        os.environ['RPPREFIX'] = os.getenv('HOME') + '/REFPROP10'
    root = os.getenv('RPPREFIX')
    print(root)
    assert(os.path.exists(root))

    RP = REFPROPFunctionLibrary(root)
    RP.SETPATHdll(root)
    # check_all_conversions(RP, FLUIDS=os.getenv('RPPREFIX')+'/FLUIDS')
    # quit()

    FLDs = sorted([os.path.split(FLD)[1].split('.')[0] for FLD in glob.glob(root+'/FLUIDS/*.FLD')])
    ref = get_all_references(FLDs)
    
    # Disabled: needed to manually fix article numbers, bugs in crossref
    # bibs = dois2bibs(list(set(ref.dois+ref.newdois)))
    # with open('FLD_bibs.bib', 'w', encoding='utf-8') as fp:
    #     fp.write(bibs)
    # cleanupbibtex('FLD_bibs.bib')

    # plot_criticals_FLD(FLD='MXYLENE', Thetamin=5e-8, Thetamax=-1e-8, deltamin=0.98, deltamax=1.02)
    # plot_criticals_FLD(FLD='CHLORINE', Thetamin=1e-6, Thetamax=-3e-7)
    # plot_criticals_FLD(FLD='DMC', Thetamin=1e-8, Thetamax=-1e-9, deltamin=0.99, deltamax=1.01)

    # test_inverse_functions()
    plot_panc_devs()
    plot_invpanc_devs()
    plot_water_nonmono(RP)
    # plot_ancillary("PROPANE")
    plot_worst()
    plot_pmu_devs()
    # plot_widths('WATER')
    
    # whyrt4()
    
    if not os.path.exists('FLD_page_cache.json'):
        cache = map_pages(['pmu_devs.pdf','devs.pdf','gooddevs.pdf','invpanc_devs.pdf'])
        with open('FLD_page_cache.json', 'w') as fp:
            fp.write(json.dumps(cache, indent=2))
    
    SI_figs = make_SI_figs(mapcache='FLD_page_cache.json')
    SI_figs_p = make_SI_figs_p(mapcache='FLD_page_cache.json')
    with open('SI_figs.tex.in', 'w') as fp:
        fp.write(SI_figs)
    with open('SI_figs_p.tex.in', 'w') as fp:
        fp.write(SI_figs_p)

    # # Test compression with brotli
    # FLD = 'WATER'
    # fname = f'output/{FLD}_exps.json'
    # if os.path.exists(fname):
    #     jj = json.load(open(fname))
    #     import brotli
    #     with open('WATER_comp.jsonbrotli', 'wb') as fp:
    #         fp.write(brotli.compress(json.dumps(jj).encode('ascii')))

    # Make table of fluid information, with check values
    make_fluid_info_table(ref)