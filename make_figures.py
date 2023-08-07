import os, timeit
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
                xticks = ax.get_xticks()
                ax.set_xticks(xticks[0:len(xticks):2])
                for ax in ax1, ax2:
                    yticks = ax.get_yticks()
                    ax.set_yticks(yticks[0:len(yticks):2])

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
            xticks = ax.get_xticks()
            ax.set_xticks(xticks[0:len(xticks):2])
            for ax in ax1, ax2:
                yticks = ax.get_yticks()
                ax.set_yticks(yticks[0:len(yticks):2])

            PDF.savefig(fig)
            plt.close()

            good_REFPROP += sum(np.isfinite(df['errP(REFPROP)']))
            bad_REFPROP += sum(~np.isfinite(df['errP(REFPROP)']))

    print(abs(bad_REFPROP/(bad_REFPROP+good_REFPROP)-1), '% of calculations fail', bad_REFPROP, good_REFPROP)

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
    warnings.filterwarnings("ignore")

    plot_ancillary("PROPANE")
    plot_worst()
    plot_pmu_devs()
    plot_widths('WATER')
    
    if not os.path.exists('FLD_page_cache.json'):
        cache = map_pages(['pmu_devs.pdf','devs.pdf','gooddevs.pdf'])
        with open('FLD_page_cache.json', 'w') as fp:
            fp.write(json.dumps(cache, indent=2))
    
    SI_figs = make_SI_figs(mapcache='FLD_page_cache.json')
    with open('SI_figs.tex.in', 'w') as fp:
        fp.write(SI_figs)

    # Test compression with brotli
    FLD = 'WATER'
    jj = json.load(open(f'output/{FLD}_exps.json'))
    import brotli
    with open('WATER_comp.jsonbrotli', 'wb') as fp:
        fp.write(brotli.compress(json.dumps(jj).encode('ascii')))