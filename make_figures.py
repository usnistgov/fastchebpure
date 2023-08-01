import os, timeit
os.environ['RPPREFIX'] = os.getenv('HOME') + '/REFPROP10'

import glob, json, os

import pandas, numpy as np, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import teqp
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary

import ChebTools 

def numprofile_stats():
    N = []
    for f in sorted(glob.glob('output/*_exps.json')):
        jL = json.load(open(f))['jexpansionsL']
        N.append(len(jL))
    print(np.std(N), np.mean(N))
numprofile_stats()
# quit()

def profile_evaluation(FLD):
    expsL, expsV = [], []
    for jL in json.load(open(f'output/{FLD}_exps.json'))['jexpansionsL']:
        eL = ChebTools.ChebyshevExpansion(jL['coef'], jL['xmin'], jL['xmax'])
        expsL.append(eL)
    for jV in json.load(open(f'output/{FLD}_exps.json'))['jexpansionsL']:
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
    for jL in json.load(open(f'output/{FLD}_exps.json'))['jexpansionsL']:
        plt.plot(jL['xmin'], jL['xmax']-jL['xmin'], 'k.')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$T_{\rm min}$ / K')
    plt.ylabel(r'$T_{\rm max}-T_{\rm min}$ / K')
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
                ax1.set_ylim(1e-16, 100)
                ax2.set_ylim(1e-16, 100)
                ax1.legend(loc='best')
                plt.suptitle(FLD)
                for ax in ax1, ax2:
                    ax.axhline(1e-12, dashes=[2,2], color='k', lw=0.5)
                    ax.axvline(1e-6, dashes=[2,2], color='k', lw=0.5)
                if ax1.get_xlim()[0] < 1e-10:
                    ax1.set_xlim(left=1e-10)
                ax1.set_ylabel(r"$|\rho'_{\Upsilon}/\rho'_{\rm ep}-1|$")
                ax2.set_ylabel(r"$|\rho''_{\Upsilon}/\rho''_{\rm ep}-1|$")
                ax2.set_xlabel(r'$\Theta\equiv (T_{\rm crit,num}-T)/T_{\rm crit,num}$')

                plt.tight_layout(pad=0.2)
                if good:
                    goodPDF.savefig(fig)
                else:
                    PDF.savefig(fig)
                plt.close()

def plot_pmu_devs():
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

            def get_VLEmu(row, rhokey):
                T = row['T / K']
                rho = row[rhokey]
                return model.get_Ar00(T, rho, z) + model.get_Ar01(T, rho, z)
            df['muVLELSA'] = df.apply(get_VLEmu, axis=1, rhokey="rho'(SA) / mol/m^3")
            df['muVLEVSA'] = df.apply(get_VLEmu, axis=1, rhokey="rho''(SA) / mol/m^3")
            df['errmu'] = np.abs(df["muVLELSA"]-df["muVLEVSA"]+np.log(df["rho'(SA) / mol/m^3"]/df["rho''(SA) / mol/m^3"]))

            ax1.plot(1-df['T / K']/Tcrit, df['errP'], lw=0.2) 
            ax2.plot(1-df['T / K']/Tcrit, df['errmu'], lw=0.2)

            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax2.set_yscale('log')
            ax1.set_ylim(1e-17, 100)
            ax2.set_ylim(1e-17, 100)
            plt.suptitle(FLD)
            for ax in ax1, ax2:
                ax.axhline(1e-12, dashes=[2,2], color='k', lw=0.5)
                ax.axvline(1e-6, dashes=[2,2], color='k', lw=0.5)
            if ax1.get_xlim()[0] < 1e-10:
                ax1.set_xlim(left=1e-10)
            ax1.set_ylabel(r"$r_p$")
            ax2.set_ylabel(r"$r_\mu$")
            ax2.set_xlabel(r'$\Theta\equiv (T_{\rm crit,num}-T)/T_{\rm crit,num}$')

            plt.tight_layout(pad=0.2)
            PDF.savefig(fig)
            plt.close()

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
    df = pandas.DataFrame(o)
    plt.plot(df['Theta'], np.abs(df['rhoLcrit']/df['rhoLmp']-1), 'o-', label='liquid(crit)', ms=1)
    plt.plot(df['Theta'], np.abs(df['rhoVcrit']/df['rhoVmp']-1), 'o-', label='vapor(crit)', dashes=[2,2], ms=1)

    plt.plot(df['Theta'], np.abs(df['rhoLconv']/df['rhoLmp']-1), 'o-', label='liquid(conv)')
    plt.plot(df['Theta'], np.abs(df['rhoVconv']/df['rhoVmp']-1), 'o-', label='vapor(conv)', dashes=[2,2])
    plt.xscale('log')
    plt.yscale('log')

    plt.gca().set(xlabel=r'$\Theta = (T_{\rm crit, num}-T)/T_{\rm crit, num}$', ylabel=r'$|\rho^{\pi}_{\rm anc}/\rho^{\pi}_{\rm ep}-1|$')
    plt.legend()
    plt.tight_layout(pad=0.2)
    plt.savefig('ancillary_boost.pdf')
    plt.show()

if __name__ == '__main__':

    root = os.getenv('RPPREFIX')
    if root == '':
        os.environ['RPPREFIX'] = os.getenv('HOME') + '/REFPROP10'
    root = os.getenv('RPPREFIX')
    print(root)
    assert(os.path.exists(root))

    RP = REFPROPFunctionLibrary(root)
    RP.SETPATHdll(root)
    
    import warnings
    # import matplotlib
    warnings.filterwarnings("ignore")

    plot_ancillary("PROPANE")
    plot_worst()
    plot_pmu_devs()
    # plot_widths('WATER')

    # plot_REFPROPdevs()

    # plot_crit()
    # plot_decrit()

    FLD = 'WATER'
    jj = json.load(open(f'output/{FLD}_exps.json'))
    import brotli
    with open('WATER_comp.jsonbrotli', 'wb') as fp:
        fp.write(brotli.compress(json.dumps(jj).encode('ascii')))