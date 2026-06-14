import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

# =========================
# 1. LOAD DATA
# =========================
metabolite_table = pd.read_excel("compartments.xlsx", sheet_name="metabolites")
rxn_df           = pd.read_excel("compartments.xlsx", sheet_name="meta-enzymes")
enzyme_df        = pd.read_excel("compartments.xlsx", sheet_name="enzymes")

# Normalize GDH names so both NAD and NADP variants map to one enzyme
gdh_aliases = {
    "glutamate dehydrogenase-nad":  "glutamate dehydrogenase",
    "glutamate dehydrogenase-nadp": "glutamate dehydrogenase",
}

rxn_df["measure"] = rxn_df["measure"].str.lower().replace(gdh_aliases)
enzyme_df["measure"] = enzyme_df["measure"].str.lower().replace(gdh_aliases)

for df in [metabolite_table, rxn_df, enzyme_df]:
    df.columns = df.columns.str.strip().str.lower()

# =========================
# 2. determine parameters
# =========================
body_mass   = 1500.0   # g, e.g. 1.5 kg juvenile
blood_mass  = body_mass * 0.06   # 6% of body mass as blood

organ_mass = {
    "muscle":  body_mass * 0.40,  # 40% of body mass as skeletal muscle
    "liver":   body_mass * 0.03,  # 3% liver
    "kidney":  body_mass * 0.005  # 0.5% both kidneys
}

# Names for acid-base metabolites in the table
H_NAME     = "H"        # free proton
HCO3_NAME  = "HCO3"     # bicarbonate
CO2_NAME   = "CO2"      # dissolved CO2
LAC_NAME   = "lactate"  # lactate anion
LACH_NAME  = "lach"     # lactic acid
NH4_NAME   = "NH4"      # ammonium
LAC_pKa    = 3.86       # pKa of lactic acid at 30°C

# Approximate blood water fraction (e.g. 80% of blood mass is water)
blood_water_fraction = 0.80
blood_water_mass = blood_mass * blood_water_fraction  # g water

T_ref = 30.0   # reference temperature (°C) 
Q10   = 2.5    # Q10
Q10_fixed = 2.5 # Q10 for fixed metabolic rate (no temp dependence)

def metabolic_rate_umolO2_min(body_mass_g, T_C, Q10_local=Q10):
    """
    Returns whole-animal metabolic rate in µmol O2/min
    using alligator SMR at 30°C and Q10 scaling.
    """
    body_mass_kg = body_mass_g / 1000.0

    # SMR at 30°C (mL O2/min), a * M^0.83
    a = 0.20  # mL O2·kg^-1·min^-1 at 30°C (scaling factor - species dependent - O2 consumption)
    SMR_mlO2_min_30 = a * (body_mass_kg ** 0.83)

    # Q10 scaling
    SMR_mlO2_min_T = SMR_mlO2_min_30 * (Q10_local ** ((T_C - T_ref) / 10.0))

    # Convert to µmol O2/min
    return SMR_mlO2_min_T * 44.6 # this is a scaling factor for the conversion from mL to µmol

T_current = 30.0  # °C the current temperature for the analyses - customisable

MR_umolO2_min = metabolic_rate_umolO2_min(body_mass, T_current)

# =========================
# 3. extract metabolites
# =========================
def extract_mets(row):
    mets = []
    for col in ["substrate1","substrate2","substrate3",
                "product1","product2","product3","product4"]:
        v = row[col]
        if isinstance(v, str) and v.lower() != "na":
            mets.append(v)
    return set(mets)

# =========================
# 4. Build the organ model
# =========================
def build_organ_model(organ_name):
    m_sub = metabolite_table[metabolite_table["compartment"] == organ_name]
    e_sub = enzyme_df[enzyme_df["compartment"] == organ_name]
    if e_sub.empty:
        return None

    rxns = []
    for _, er in e_sub.iterrows():
        enz = er["measure"]
        act_mean = er["activity"]
        se       = er["se"]
        n        = er["n"]

        rrow = rxn_df[rxn_df["measure"] == enz]
        if rrow.empty:
            continue
        r = rrow.iloc[0]

        subs = [r["substrate1"], r["substrate2"], r["substrate3"]]
        subs = [s for s in subs if isinstance(s, str) and s.lower() != "na"]
        prods = [r["product1"], r["product2"], r["product3"], r["product4"]]
        prods = [p for p in prods if isinstance(p, str) and p.lower() != "na"]

        if not subs:
            continue

        reverse_flag  = str(r.get("reverse", "no")).strip().lower()
        is_reversible = (reverse_flag == "yes")

        rxns.append({
            "enzyme": enz,
            "act_mean": act_mean,
            "se": se,
            "n": n,
            "subs": subs,
            "prods": prods,
            "reversible": is_reversible,
            "primary": None,
        }) 
        
    mets = set()
    for r in rxns:
        mets.update(r["subs"])
        mets.update(r["prods"])

    for acid_met in [H_NAME, HCO3_NAME, CO2_NAME, LAC_NAME, LACH_NAME, NH4_NAME]:
        mets.add(acid_met)

    metabolites = sorted(mets)
    met_idx = {m: i for i, m in enumerate(metabolites)}

    blood_rows = metabolite_table[metabolite_table["compartment"] == "blood"]

    # physiological fallback (only used if blood ALSO missing)
    fallback = {
        H_NAME:    (10**(-7.08) / 1060.0) * 1e6,
        HCO3_NAME: 0.0105,      # 10.5 mM
        CO2_NAME:  0.0012,      # 1.2 mM
        LAC_NAME:  1.0e-3,      # 1 mM lactate fallback
        LACH_NAME: 0.0          # negligible lactic acid
    }

    C0 = np.zeros(len(metabolites))

    for m, i in met_idx.items():

        # 1) Try organ value
        organ_rows_m = m_sub[m_sub["measure"] == m]
        if not organ_rows_m.empty:
            val = organ_rows_m["mean"].iloc[0]
        else:
            val = np.nan

        # 2) If missing, use BLOOD value
        if (not np.isfinite(val)) or (val is None):
            blood_rows_m = blood_rows[blood_rows["measure"] == m]
            if not blood_rows_m.empty:
                val = blood_rows_m["mean"].iloc[0]

        # 3) If still missing, use fallback
        if (not np.isfinite(val)) or (val is None):
            val = fallback.get(m, 0.0)

        # 4) Enforce positivity for acid–base species
        if m in [H_NAME, HCO3_NAME, CO2_NAME]:
            if val <= 0:
                val = fallback[m]

        C0[i] = float(val)

    # Final safety
    bad = ~np.isfinite(C0)
    if bad.any():
        C0[bad] = 0.0

    # Build concentration map
    conc_map = {m: C0[met_idx[m]] for m in metabolites}

    for r in rxns:
        subs = r["subs"]
        known_subs = [(s, conc_map[s]) for s in subs if s in conc_map]
        if known_subs:
            primary = min(known_subs, key=lambda x: x[1])[0]
        else:
            primary = subs[0]
        r["primary"] = primary

    def odes_placeholder(t, y):
        raise RuntimeError("Use sampled organ model with k_f/k_r (forward and reverse reactions), not the base template.")

    return {
        "organ": organ_name,
        "metabolites": metabolites,
        "met_idx": met_idx,
        "C0": C0,
        "rxns": rxns,
        "conc_map": conc_map,
        "odes": odes_placeholder,
    }

# =========================
# 5. LOAD DATA
# =========================
def sample_organ_model(base_org, alpha=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    met_idx   = base_org["met_idx"]
    conc_map  = base_org["conc_map"]
    organ_name = base_org["organ"]

    rxns_sampled = []

    for r in base_org["rxns"]:
        act_mean = r["act_mean"]
        se       = r["se"]
        n        = r["n"]

        sd = se * np.sqrt(n) if (pd.notna(se) and pd.notna(n) and n > 0) else 0.0
        a = rng.normal(act_mean, sd)
        a = max(a, 0.0)

        primary    = r["primary"]
        subs       = r["subs"]
        prods      = r["prods"]
        reversible = r["reversible"]

        C0_primary = conc_map.get(primary, 1e-6)
        if (not np.isfinite(C0_primary)) or (C0_primary <= 0):
            C0_primary = 1e-6
        k_f = a / C0_primary

        if reversible:
            prod_concs = [conc_map[p] for p in prods if p in conc_map]
            P0 = np.prod(prod_concs) if prod_concs else 1e-6
            if (not np.isfinite(P0)) or (P0 <= 0):
                k_r = 0.0
            else:
                k_r = alpha * a / P0
        else:
            k_r = 0.0

        rxns_sampled.append({
            "k_f": k_f,
            "k_r": k_r,
            "subs": subs,
            "prods": prods,
            "primary": primary,
            "reversible": reversible,
        })

    # =========================
    # 6. build ODE (physiological model with mass-action kinetics + acid-base equilibria)
    # =========================
    def odes(t, y):
        y = np.where(np.isfinite(y), y, 0.0)
        y = np.maximum(y, 0.0)

        d = np.zeros_like(y)

        # mass-action fluxes
        for r in rxns_sampled:
            v_f = r["k_f"]
            for s in r["subs"]:
                ys = y[met_idx[s]]
                if (not np.isfinite(ys)) or (ys <= 0):
                    ys = 0.0
                v_f *= ys

            v_r = 0.0
            if r["reversible"] and r["k_r"] > 0.0:
                v_r = r["k_r"]
                for p in r["prods"]:
                    yp = y[met_idx[p]]
                    if (not np.isfinite(yp)) or (yp <= 0):
                        yp = 0.0
                    v_r *= yp

            v = v_f - v_r

            for s in r["subs"]:
                d[met_idx[s]] -= v
            for p in r["prods"]:
                d[met_idx[p]] += v

        # CO2/HCO3 equilibrium
        if (H_NAME in met_idx) and (HCO3_NAME in met_idx) and (CO2_NAME in met_idx):
            i_H    = met_idx[H_NAME]
            i_HCO3 = met_idx[HCO3_NAME]
            i_CO2  = met_idx[CO2_NAME]

            HCO3_conc = max(y[i_HCO3], 1e-12) # to adjust for right measurements
            CO2_conc  = max(y[i_CO2], 1e-12) # to adjust for right measurements

            # H_conc is in µmol/g
            H_umol_per_g = max(y[i_H], 1e-12) # to adjust for right measurements

            # convert to mol/L
            H_mol_per_L = (H_umol_per_g * 1e-6) * 1060.0   # 1060 g blood per L

            # compute pH
            pH = -np.log10(max(H_mol_per_L, 1e-20)) # to adjust for right measurements
            pK_prime = -0.0817 * pH + 6.7818 # to adjust for the right measurements/ constants for disassociation of protons

            C_T = HCO3_conc + CO2_conc

            if C_T > 0.0 and np.isfinite(pH) and np.isfinite(pK_prime):
                ratio = 10.0 ** (pH - pK_prime)
                CO2_eq  = C_T / (1.0 + ratio)
                HCO3_eq = C_T - CO2_eq

                CO2_eq  = max(CO2_eq, 1e-12) # to adjust for the right measurements 
                HCO3_eq = max(HCO3_eq, 1e-12) # to adjust for the right measurements

                y[i_CO2]  = CO2_eq
                y[i_HCO3] = HCO3_eq

                d[i_CO2]  = 0.0
                d[i_HCO3] = 0.0

        # Lactic acid <-> lactate + H+ equilibrium
        if (H_NAME in met_idx) and (LAC_NAME in met_idx) and (LACH_NAME in met_idx):
            i_H    = met_idx[H_NAME]
            i_LAC  = met_idx[LAC_NAME]
            i_LACH = met_idx[LACH_NAME]

            # H_conc is in µmol/g
            H_umol_per_g = max(y[i_H], 1e-12)

            # convert to mol/L
            H_mol_per_L = (H_umol_per_g * 1e-6) * 1060.0   # 1060 g blood per L

            # compute pH
            pH = -np.log10(max(H_mol_per_L, 1e-20))

            LAC_conc  = max(y[i_LAC], 0.0)
            LACH_conc = max(y[i_LACH], 0.0)

            L_tot = LAC_conc + LACH_conc

            # this is for the total lactate concentration in µmol/g, which is what we are trying to maintain at equilibrium
            if L_tot > 0.0 and np.isfinite(pH):
                ratio = 10.0 ** (pH - LAC_pKa)
                LAC_eq  = L_tot * ratio / (1.0 + ratio)
                LACH_eq = L_tot - LAC_eq

                LAC_eq  = max(LAC_eq, 0.0)
                LACH_eq = max(LACH_eq, 0.0)

                y[i_LAC]  = LAC_eq
                y[i_LACH] = LACH_eq

                d[i_LAC]  = 0.0
                d[i_LACH] = 0.0

        d = np.where(np.isfinite(d), d, 0.0)
        return d

    return {
        "organ": base_org["organ"],
        "metabolites": base_org["metabolites"],
        "met_idx": met_idx,
        "C0": base_org["C0"].copy(),
        "odes": odes,
    }

# =========================
# 7. Build the sample organ models for all organs (with Monte-Carlo sampling)
# =========================
def build_sampled_organs(rng=None):
    if rng is None:
        rng = np.random.default_rng()
    organs_mc = {}
    for name, base_org in organs.items():
        organs_mc[name] = sample_organ_model(base_org, rng=rng)
    return organs_mc

# =========================
# 8. Test run one organ to equilibrium (kidney)
# =========================
organs = {}

for organ_name in metabolite_table["compartment"].unique():
    model = build_organ_model(organ_name)
    if model is not None:
        organs[organ_name] = model
        print("Built organ:", organ_name)

if len(organs) == 0:
    raise RuntimeError("No organs were built — check your Excel sheet names.")

# Example: run one organ to equilibrium
if "kidney" in organs:
    rng = np.random.default_rng(42)
    kidney_model = sample_organ_model(organs["kidney"], rng=rng)
    y0 = kidney_model["C0"]
    t_span = (0.0, 1000.0)
    sol = solve_ivp(kidney_model["odes"], t_span, y0, method="LSODA", dense_output=False)

    print("Final concentrations (kidney):")
    for m, i in kidney_model["met_idx"].items():
        print(m, sol.y[i, -1])

# =========================
# 9. Define active organs and metabolic rate
# =========================
# All organs except blood are active; blood is the shared compartment
ACTIVE_ORGANS = [o for o in organs.keys() if o != "blood"]

# Baseline metabolic acid load into blood (µmol/(g_blood·min))
baseline_acid_load = 0.0  # set >0 if you want a constant acid load

def metabolic_rate_umolO2_min(body_mass_g, T_C, Q10):
    body_mass_kg = body_mass_g / 1000.0
    a = 0.20
    SMR_mlO2_min_30 = a * (body_mass_kg ** 0.83)
    SMR_mlO2_min_T = SMR_mlO2_min_30 * (Q10 ** ((T_C - T_ref) / 10.0))
    return SMR_mlO2_min_T * 44.6   # convert mL O2/min → µmol O2/min

def metabolic_rates(body_mass_g, T_C, Q10, RQ=0.85):
    VO2 = metabolic_rate_umolO2_min(body_mass_g, T_C, Q10)
    VCO2 = VO2 * RQ
    return VO2, VCO2

# =========================
# 10. This builds the global model between all organs 
# =========================
def build_global_model_with_orgs(
        active_orgs,
        organs_mc,
        lac_rate_per_g=0.0,
        nh4_rate_per_g=0.0,
        baseline_acid_load=0.0,
        k_excr_NH4=0.0,
        k_CA=0.0,
        acid_factor=1.0,
        gamma=1.7,
        D=0.0,
        eta_Asp=0.0,
        gamma_Asp=0.0
    ):
    """
    active_orgs: list of organ names (excluding blood)
    organs_mc: sampled organ models
    lac_rate_per_g, nh4_rate_per_g: whole-animal metabolic loads
    baseline_acid_load: constant H+ load into blood
    """

    org_names = active_orgs
    n_org = len(org_names)
    
    all_mets = sorted({m for o in org_names for m in organs_mc[o]["metabolites"]})
    met_idx_global = {m: i for i, m in enumerate(all_mets)}
    n_m = len(all_mets)

    def build_C0():
        y0_parts = []

        # Organ compartments
        for o in org_names:
            org = organs_mc[o]
            vec = np.zeros(n_m)
            for m, i_local in org["met_idx"].items():
                vec[met_idx_global[m]] = org["C0"][i_local]
            y0_parts.append(vec)

        # Blood starts as average of organs (acid baseline already applied)
        blood0 = sum(y0_parts) / len(y0_parts)
        blood0[met_idx_global["aspartate"]] = 0.0
        y0_parts.append(blood0)
        return np.concatenate(y0_parts)

    def global_odes(t, y):
        
        # Clamp state - to limit unbelievable values (eg less than 0 or non-finite)
        y = np.where(np.isfinite(y), y, 0.0)
        y = np.maximum(y, 0.0)

        # Slice into organ compartments + blood
        org_states = [y[i*n_m:(i+1)*n_m] for i in range(n_org)]
        blood      = y[n_org*n_m:(n_org+1)*n_m]

        d_org   = [np.zeros(n_m) for _ in org_names]
        d_blood = np.zeros(n_m)

        # -----------------------------------------------------
        # A. INTERNAL ORGAN METABOLISM (ALL ENZYMES)
        # -----------------------------------------------------
        for oi, o in enumerate(org_names):
            org = organs_mc[o]
            d_local = org["odes"](t, org_states[oi])
            d_org[oi] += d_local

        # -----------------------------------------------------
        # B. DIFFUSIVE EXCHANGE WITH BLOOD
        # -----------------------------------------------------
        k = 0.05  # min^-1
        for oi, o in enumerate(org_names):
            flux = k * (blood - org_states[oi])
            d_org[oi] += flux
            d_blood  -= flux
            
        # -----------------------------------------------------
        # BASELINE ACID LOAD (I did not use but can if want to tweak the model)
        # -----------------------------------------------------
        H_idx = met_idx_global.get(H_NAME, None)
        if H_idx is not None and baseline_acid_load != 0.0:
            d_blood[H_idx] += baseline_acid_load

        # -----------------------------------------------------
        # C. WHOLE-ANIMAL METABOLIC LOADS
        # -----------------------------------------------------
        lac_idx = met_idx_global.get(LAC_NAME, None)

        nh4_idx = met_idx_global.get(NH4_NAME, None)
        
        if H_idx is not None:
            effective_acid_load = acid_factor * (lac_rate_per_g ** gamma)
            d_blood[H_idx] += effective_acid_load


        # ---------------------------------------------------------
        # WHOLE-ANIMAL METABOLIC RATES (O2 and CO2)
        # ---------------------------------------------------------
        VO2_umol_min, VCO2_umol_min = metabolic_rates(
            body_mass_g=body_mass,
            T_C=T_current,
            Q10=Q10_fixed,
            RQ=0.85
        )

        # -----------------------------------------------------
        # D. WHOLE-ANIMAL CO2 PRODUCTION
        # -----------------------------------------------------
        CO2_idx = met_idx_global.get(CO2_NAME, None)
        if CO2_idx is not None:
            d_blood[CO2_idx] += VCO2_umol_min / blood_mass

        # -----------------------------------------------------
        # E. CARBONIC ANHYDRASE (CO2 ↔ H+)
        # -----------------------------------------------------
        if CO2_idx is not None and H_idx is not None:
            CO2_umol_per_g = blood[CO2_idx]
            H_umol_per_g   = blood[H_idx]

            CO2_mol_per_L = CO2_umol_per_g * 1e-6 * 1060.0
            H_mol_per_L   = H_umol_per_g   * 1e-6 * 1060.0

            Keq  = 10**(-6.1)
            # carbonic anhydrase
            v_CA_mol_per_L = k_CA * (CO2_mol_per_L - H_mol_per_L / Keq)

            v_CA_umol_per_g = v_CA_mol_per_L / 1060.0 * 1e6

            d_blood[CO2_idx] -= v_CA_umol_per_g
            d_blood[H_idx]   += v_CA_umol_per_g

        # -----------------------------------------------------
        # Renal NH4HCO3 excretion (equimolar NH4+ and HCO3- removal)
        # -----------------------------------------------------
        HCO3_idx = met_idx_global.get(HCO3_NAME, None)

        ASP_idx = met_idx_global["aspartate"]
        ASP = blood[ASP_idx]

        k_clear_ASP = 0.01  # umol/g/min^-1, clearance rate for aspartate (physiologically reasonable but can be tuned)
        
        # infusion + clearance
        dASP_dt = aspartate_infusion(t, D) - k_clear_ASP * ASP
        d_blood[ASP_idx] += dASP_dt

        # modify NH4 sink
        if "NH4" in met_idx_global:
            nh4_idx = met_idx_global["NH4"]
            NH4_blood = blood[nh4_idx]
            k_excr_eff = k_excr_NH4 * (1 + eta_Asp * ASP)
            J_excr_NH4 = k_excr_eff * NH4_blood * blood_mass
            d_blood[nh4_idx] -= J_excr_NH4 / blood_mass

        # modify anaerobic fraction (if used inside ODE)
        # anaer_frac_eff = anaer_frac_base / (1 + gamma_Asp * ASP)

        # -----------------------------------------------------
        # E. CLEANUP
        # -----------------------------------------------------
        for oi in range(n_org):
            d_org[oi] = np.where(np.isfinite(d_org[oi]), d_org[oi], 0.0)
        d_blood = np.where(np.isfinite(d_blood), d_blood, 0.0)

        # Concatenate output
        out = []
        for oi in range(n_org):
            out.append(d_org[oi])
        out.append(d_blood)

        return np.concatenate(out)

    return global_odes, build_C0, met_idx_global

# =========================
# 11. This is to reset the blood equilibruim to be consistent with physiology and to remap the pH and metabolites throughout the simulation
# =========================
def reset_blood_lactate_equilibrium(y0, met_idx_global, L_tot_umol_per_g, pH0):
    n_m = len(met_idx_global)
    blood = y0[-n_m:]

    lac_idx  = met_idx_global[LAC_NAME]
    lach_idx = met_idx_global[LACH_NAME]

    ratio = 10**(pH0 - LAC_pKa)
    LAC_eq  = L_tot_umol_per_g * ratio / (1 + ratio)
    LACH_eq = L_tot_umol_per_g - LAC_eq

    blood[lac_idx]  = LAC_eq
    blood[lach_idx] = LACH_eq

    y0[-n_m:] = blood
    return y0

# =========================
# 12. Hyperparameter tuning for rest calibration
# =========================
# NH4HCO3 renal excretion parameters
NH4_EXCR_TOTAL_REST = 6.1   # µmol/min for 1.5 kg animal (scaled from 4.3 µmol/min at 1.05 kg)

NH4_REST_CONC = 2.0         # µmol/g, choose a plausible resting NH4+ concentration
blood_mass_g   = blood_mass # you already have this

# clearance constant k_excr (min^-1)
k_excr_NH4 = NH4_EXCR_TOTAL_REST / (NH4_REST_CONC * blood_mass_g)

# =========================
# 13. Monte Carlo simulation for the rest calibration and parameter optimisation
# =========================
def sample_organs_mc(organs, rng):
    """
    Monte‑Carlo sampling of alligator organs.
    Returns organs_mc: a dict of organ models with sampled masses & C0.
    """
    organs_mc = {}
    for name, base_org in organs.items():
        if name == "blood":
            continue
        organs_mc[name] = sample_organ_model(base_org, rng=rng)
    return organs_mc

def initialise_blood_state(build_C0, met_idx_global,
                           L_tot0=6.095, pH0=7.08):
    """
    Builds initial condition vector and sets lactate/pH equilibrium.
    """
    y0 = build_C0()
    y0 = reset_blood_lactate_equilibrium(y0, met_idx_global, L_tot0, pH0)
    return y0

# now for aspartate infusion during exertion simulation - note only lasts 60 minutes can be tweaked with t=60
def aspartate_infusion(t, D, t_start=0.0, t_end=60.0):
    """
    Returns infusion rate (µmol/g/min) during IV infusion window.
    """
    if t_start <= t <= t_end:
        return D
    return 0.0

def build_alligator_model(organs_mc,
                          lac_rate_per_g,
                          nh4_rate_per_g,
                          k_excr_NH4,
                          k_CA,
                          acid_factor=1.0,
                          gamma=1.7,
                          D=0.0,
                          eta_Asp=0.0,
                          gamma_Asp=0.0):
    """
    Wrapper for build_global_model_with_orgs.
    Passes ASP parameters through.
    """
    global_odes, build_C0, met_idx_global = build_global_model_with_orgs(
        ACTIVE_ORGANS,
        organs_mc,
        lac_rate_per_g=lac_rate_per_g,
        nh4_rate_per_g=nh4_rate_per_g,
        baseline_acid_load=0.0,
        k_excr_NH4=k_excr_NH4,
        k_CA=k_CA,
        acid_factor=acid_factor,
        gamma=gamma,
        D=D,
        eta_Asp=eta_Asp,
        gamma_Asp=gamma_Asp
    )
    return global_odes, build_C0, met_idx_global

# =========================
# 14. Now run the rest model
# =========================
def run_rest_model(acid_factor, k_excr_NH4, k_CA):
    """
    Runs the model at anaer_frac = 0.05 (rest calibration).
    Returns final blood state and met_idx.
    """
    anaer_frac = 0.05 # alter for the fraction of metabolism that is anaerobic at rest (can be tuned)

    MR = metabolic_rate_umolO2_min(body_mass, T_current, Q10_fixed)
    anaerobic_rate = MR * anaer_frac

    lac_total_rate = (2.0/6.0) * anaerobic_rate # this is 2 lactate for every glucose = 6 ATP molecules/ 2 lactose 
    nh4_total_rate = 0.25 * lac_total_rate # arbitrairy rate of NH4 to lactate production - gets tuned later 
    
    lac_rate_per_g = lac_total_rate / blood_mass
    nh4_rate_per_g = nh4_total_rate / blood_mass

    rng = np.random.default_rng(12345)
    organs_mc = sample_organs_mc(organs, rng)

    global_odes, build_C0, met_idx_global = build_alligator_model(
        organs_mc,
        lac_rate_per_g,
        nh4_rate_per_g,
        k_excr_NH4,
        k_CA,
        acid_factor
    )

    y0 = initialise_blood_state(build_C0, met_idx_global)
    sol = solve_ivp(global_odes, (0, 240), y0, method="BDF")

    n_m = len(met_idx_global)
    blood_final = sol.y[-n_m:, -1]
    return blood_final, met_idx_global


def objective_rest(theta):
    acid_factor, k_excr_NH4, k_CA = theta
    blood_final, met_idx = run_rest_model(acid_factor, k_excr_NH4, k_CA)

    H_idx   = met_idx[H_NAME]
    LAC_idx = met_idx[LAC_NAME]
    LACH_idx = met_idx[LACH_NAME]
    NH4_idx = met_idx[NH4_NAME]

    H    = blood_final[H_idx]
    LAC  = blood_final[LAC_idx]
    LACH = blood_final[LACH_idx]
    NH4  = blood_final[NH4_idx]

    H_mol_L = H * 1e-6 * 1060.0
    pH_final = -np.log10(max(H_mol_L, 1e-20))

    # weights - generated by the se from spreadsheet
    w_pH  = 1.0 / (0.04**2)
    w_LAC = 1.0 / (0.49**2)
    w_NH4 = 1.0 / (0.03**2)

    # CA prior - from spreadsheet
    k_CA_prior = 0.0039
    SE_CA = 0.00077
    J_CA = ((k_CA - k_CA_prior) / SE_CA)**2

    # these are prior numbers
    J = (
        w_pH  * (pH_final - 7.08)**2 +
        w_LAC * ((LAC + LACH) - 6.095)**2 +
        w_NH4 * (NH4 - 0.205)**2 +
        J_CA
    )
    return J


def optimise_rest_params():
    theta0 = [1.0, k_excr_NH4, 0.0039]
    bounds = [(0, 5), (1e-6, 1), (1e-5, 1)]
    res = minimize(objective_rest, theta0, bounds=bounds)
    return res.x

best_theta = optimise_rest_params()
acid_factor_opt = best_theta[0]
k_excr_NH4_opt  = best_theta[1]
k_CA_opt        = best_theta[2]

# =========================
# 15. Hyperparameter globalised self-tuning
# =========================
def anaerobic_feedback(H, CO2, H_start, CO2_start,
                       alpha, beta, k, theta):
    H_mol_L = H * 1e-6 * 1060.0
    H_rel   = H_mol_L - (H_start * 1e-6 * 1060.0)
    CO2_rel = CO2 - CO2_start
    signal = alpha * H_rel + beta * CO2_rel
    return 1.0 / (1.0 + np.exp(-k * (signal - theta)))


def run_model_feedback(alpha, beta, k, theta,
                       acid_factor_opt, k_excr_NH4_opt, k_CA_opt):
    """
    Runs the model with feedback controlling anaerobic fraction.
    """
    rng = np.random.default_rng(999)
    organs_mc = sample_organs_mc(organs, rng)

    global_odes, build_C0, met_idx_global = build_alligator_model(
        organs_mc,
        lac_rate_per_g=0.0,
        nh4_rate_per_g=0.0,
        k_excr_NH4=k_excr_NH4_opt,
        k_CA=k_CA_opt,
        acid_factor=acid_factor_opt
    )

    y0 = initialise_blood_state(build_C0, met_idx_global)
    sol = solve_ivp(global_odes, (0, 240), y0, method="BDF")

    blood_block = sol.y[-len(met_idx_global):, :]
    H_final   = blood_block[met_idx_global[H_NAME], -1]
    CO2_final = blood_block[met_idx_global[CO2_NAME], -1]

    H_start = y0[met_idx_global[H_NAME]]
    CO2_start = y0[met_idx_global[CO2_NAME]]

    f_ana = anaerobic_feedback(
        H_final, CO2_final,
        H_start, CO2_start,
        alpha, beta, k, theta
    )

    anaer_frac_eff = 0.5 * f_ana # adjust for the anaerobic rate required
    return blood_block[:, -1], anaer_frac_eff, met_idx_global, y0


def objective_feedback(theta_vec):
    alpha, beta, k, theta = theta_vec

    blood_final, _, met_idx, y0 = run_model_feedback(
        alpha, beta, k, theta,
        acid_factor_opt, k_excr_NH4_opt, k_CA_opt
    )

    H_idx    = met_idx[H_NAME]
    LAC_idx  = met_idx[LAC_NAME]
    LACH_idx = met_idx[LACH_NAME]
    NH4_idx  = met_idx[NH4_NAME]

    H_start    = y0[H_idx]
    LAC_start  = y0[LAC_idx]
    LACH_start = y0[LACH_idx]
    NH4_start  = y0[NH4_idx]

    H_final    = blood_final[H_idx]
    LAC_final  = blood_final[LAC_idx]
    LACH_final = blood_final[LACH_idx]
    NH4_final  = blood_final[NH4_idx]

    H_mol_L_final = H_final * 1e-6 * 1060.0
    pH_final = -np.log10(max(H_mol_L_final, 1e-20))

    H_mol_L_start = H_start * 1e-6 * 1060.0
    pH_start = -np.log10(max(H_mol_L_start, 1e-20))

    J = (
        (pH_final - pH_start)**2 +
        ((LAC_final + LACH_final) - (LAC_start + LACH_start))**2 +
        (NH4_final - NH4_start)**2
    )
    return J


def optimise_feedback_params():
    bounds = [(0,10), (0,10), (0.1,10), (-5,5)]
    result = differential_evolution(objective_feedback, bounds)
    return result.x

# =========================
# 16. Exertional modelling
# =========================
def simulate_exertion_levels(acid_factor_opt, k_excr_NH4_opt, k_CA_opt):
    results = {name: {} for name in EXERTION_LEVELS}

    for ex_name, anaer_frac in EXERTION_LEVELS.items():
        traces = {"lactate": [], "lactic_acid": [], "NH4": [], "pH": []}

        for mc in range(N_MC):
            rng = np.random.default_rng(1000 + mc)
            organs_mc = sample_organs_mc(organs, rng)

            MR = metabolic_rate_umolO2_min(body_mass, T_current, Q10_fixed)
            anaerobic_rate = MR * anaer_frac

            lac_total_rate = (2.0/6.0) * anaerobic_rate
            nh4_total_rate = 0.25 * lac_total_rate

            lac_rate_per_g = lac_total_rate / blood_mass
            nh4_rate_per_g = nh4_total_rate / blood_mass

            global_odes, build_C0, met_idx_global = build_alligator_model(
                organs_mc,
                lac_rate_per_g,
                nh4_rate_per_g,
                k_excr_NH4_opt,
                k_CA_opt,
                acid_factor_opt
            )

            y0 = initialise_blood_state(build_C0, met_idx_global)
            sol = solve_ivp(global_odes, (0, t_end), y0, method="BDF", t_eval=time)

            blood_block = sol.y[-len(met_idx_global):, :]
            H_trace = blood_block[met_idx_global[H_NAME], :]
            H_mol_L = H_trace * 1e-6 * 1060.0
            pH_trace = -np.log10(np.maximum(H_mol_L, 1e-20))

            traces["lactate"].append(blood_block[met_idx_global[LAC_NAME], :])
            traces["lactic_acid"].append(blood_block[met_idx_global[LACH_NAME], :])
            traces["NH4"].append(blood_block[met_idx_global[NH4_NAME], :])
            traces["pH"].append(pH_trace)

        results[ex_name] = traces

    return results


def run_model_with_aspartate(D, eta_Asp, gamma_Asp):
    
    rng = np.random.default_rng(777)
    organs_mc = sample_organs_mc(organs, rng)

    global_odes, build_C0, met_idx_global = build_alligator_model(
        organs_mc,
        lac_rate_per_g=0.0,
        nh4_rate_per_g=0.0,
        k_excr_NH4=k_excr_NH4_opt,
        k_CA=k_CA_opt,
        acid_factor=acid_factor_opt,
        gamma=1.7,
        D=D,
        eta_Asp=eta_Asp,
        gamma_Asp=gamma_Asp
    )

    y0 = initialise_blood_state(build_C0, met_idx_global)
    sol = solve_ivp(global_odes, (0, 240), y0, method="BDF")

    blood_final = sol.y[-len(met_idx_global):, -1]

    # compute effective anaerobic fraction
    ASP = blood_final[met_idx_global["aspartate"]]
    anaer_frac_eff = 0.5 / (1 + gamma_Asp * ASP)

    return blood_final, y0, met_idx_global, anaer_frac_eff

rng = np.random.default_rng(2026)
organs_mc = sample_organs_mc(organs, rng)

# =========================
# 17. Modelling the effects of aspartate
# =========================
def objective_aspartate(theta):
    D, eta_Asp, gamma_Asp = theta

    blood_final, y0, met_idx, anaer_frac_eff = run_model_with_aspartate(
        D, eta_Asp, gamma_Asp
    )

    H_idx    = met_idx[H_NAME]
    LAC_idx  = met_idx[LAC_NAME]
    LACH_idx = met_idx[LACH_NAME]
    NH4_idx  = met_idx[NH4_NAME]

    # starting state
    H_start    = y0[H_idx]
    LAC_start  = y0[LAC_idx]
    LACH_start = y0[LACH_idx]
    NH4_start  = y0[NH4_idx]

    # final state
    H_final    = blood_final[H_idx]
    LAC_final  = blood_final[LAC_idx]
    LACH_final = blood_final[LACH_idx]
    NH4_final  = blood_final[NH4_idx]

    # pH
    pH_start = -np.log10((H_start * 1e-6 * 1060.0))
    pH_final = -np.log10((H_final * 1e-6 * 1060.0))

    # objective: minimise anaerobic fraction + deviation from starting state
    J = (
        10.0 * anaer_frac_eff**2 +   # strong penalty on anaerobic metabolism
        (pH_final - pH_start)**2 +
        ((LAC_final + LACH_final) - (LAC_start + LACH_start))**2 +
        (NH4_final - NH4_start)**2
    )
    return J
def optimise_aspartate_params():
    bounds = [
        (0, 50),   # D: dose
        (0, 5),    # eta_Asp = effect of aspartate in NH4 excretion
        (0, 5)     # gamma_Asp = effect of aspartate on aerobic respiration
    ]
    result = differential_evolution(objective_aspartate, bounds)
    return result.x

best_D, best_eta_Asp, best_gamma_Asp = optimise_aspartate_params()
print("Optimal aspartate parameters:", best_D, best_eta_Asp, best_gamma_Asp)

global_odes, build_C0, met_idx_global = build_alligator_model(
    organs_mc,
    lac_rate_per_g=0.0,
    nh4_rate_per_g=0.0,
    k_excr_NH4=k_excr_NH4_opt,
    k_CA=k_CA_opt,
    acid_factor=acid_factor_opt,
    gamma=1.7,
    D=best_D,
    eta_Asp=best_eta_Asp,
    gamma_Asp=best_gamma_Asp
)

y0 = initialise_blood_state(build_C0, met_idx_global)

t_end = 240.0
time = np.arange(0, t_end + 1e-9, 10.0)

sol = solve_ivp(global_odes, (0, t_end), y0, method="BDF", t_eval=time)

n_m = len(met_idx_global)
blood_block = sol.y[-n_m:, :]

LAC = blood_block[met_idx_global[LAC_NAME], :]
LACH = blood_block[met_idx_global[LACH_NAME], :]
L_total = LAC + LACH

H_trace = blood_block[met_idx_global[H_NAME], :]
H_mol_L = H_trace * 1e-6 * 1060.0
pH_trace = -np.log10(np.maximum(H_mol_L, 1e-20))

# =========================
# 18. Run the final hyperparameterisation model
# =========================
def run_model(anaer_frac, acid_factor, k_excr_NH4, k_CA):
    """
    Runs the full ODE model for a given anaerobic fraction and parameters.
    Returns blood_final (vector of metabolite concentrations).
    """

    # 1. metabolic rate
    MR_umolO2_min = metabolic_rate_umolO2_min(body_mass, T_current, Q10_fixed)
    anaerobic_rate = MR_umolO2_min * anaer_frac

    lac_total_rate = (2.0/6.0) * anaerobic_rate
    nh4_total_rate = 0.25 * lac_total_rate

    lac_rate_per_g = lac_total_rate / blood_mass
    nh4_rate_per_g = nh4_total_rate / blood_mass

    # acid load scaling
    effective_acid_load = acid_factor * (anaer_frac ** 1.7)

    # 2. sample organs
    rng = np.random.default_rng(12345)
    organs_mc = {
        name: sample_organ_model(base_org, rng=rng)
        for name, base_org in organs.items()
        if name != "blood"
    }

    # 3. build model with parameters
    global_odes, build_C0, met_idx_global = build_global_model_with_orgs(
        ACTIVE_ORGANS,
        organs_mc,
        lac_rate_per_g=lac_rate_per_g,
        nh4_rate_per_g=nh4_rate_per_g,
        baseline_acid_load=0.0,
        k_excr_NH4=k_excr_NH4,
        k_CA=k_CA,
        acid_factor=acid_factor,
        gamma=1.7
    )

    # 4. initial conditions
    y0 = build_C0()
    y0 = reset_blood_lactate_equilibrium(y0, met_idx_global, 6.095, 7.08)

    # 5. solve
    sol = solve_ivp(global_odes, (0, 240), y0, method="BDF")

    # 6. extract final blood state
    n_m = len(met_idx_global)
    blood_final = sol.y[-n_m:, -1]

    return blood_final, met_idx_global

def simulate_rest(theta):
    acid_factor, k_excr_NH4, k_CA = theta
    blood_final, met_idx = run_model(
        anaer_frac=1.00,
        acid_factor=acid_factor,
        k_excr_NH4=k_excr_NH4,
        k_CA=k_CA
    )
    return blood_final, met_idx

def objective(theta):
    
    blood_final, met_idx = simulate_rest(theta)

    # extract indices
    H_idx   = met_idx[H_NAME]
    LAC_idx = met_idx[LAC_NAME]
    LACH_idx = met_idx[LACH_NAME]
    NH4_idx = met_idx[NH4_NAME]

    # extract values
    H    = blood_final[H_idx]
    LAC  = blood_final[LAC_idx]
    LACH = blood_final[LACH_idx]
    NH4  = blood_final[NH4_idx]

    # compute pH
    H_mol_per_L = H * 1e-6 * 1060.0
    pH_final = -np.log10(max(H_mol_per_L, 1e-20))

    # weights from SE
    SE_pH  = 0.04
    SE_LAC = 0.49
    SE_NH4 = 0.03

    w_pH  = 1.0 / (SE_pH**2)
    w_LAC = 1.0 / (SE_LAC**2)
    w_NH4 = 1.0 / (SE_NH4**2)

    # CA regularization
    k_CA_prior = 0.0039
    SE_CA      = 0.00077
    k_CA       = theta[2]

    J_CA = ((k_CA - k_CA_prior) / SE_CA)**2

    # main cost
    J = (
        w_pH  * (pH_final - 7.08)**2 +
        w_LAC * ((LAC + LACH) - 6.095)**2 +
        w_NH4 * (NH4 - 0.205)**2 +
        J_CA
    )

    return J

theta0 = [1.0, k_excr_NH4, 0.0039]
bounds = [(0, 5), (1e-6, 1), (1e-5, 1)]

res = minimize(objective, theta0, bounds=bounds)
print("Optimized parameters (NH4 excretion rate):", res.x)

best_theta = res.x
acid_factor_opt = best_theta[0]
k_excr_NH4_opt  = best_theta[1]
k_CA_opt        = best_theta[2]

# =========================
# 19. Monte Carlo simulation (final)
# =========================
EXERTION_LEVELS = {
    "rest":    0.05,   # 5% anaerobic
    "mild":    0.10,   # 10% anaerobic
    "burst":   0.50,   # 50% anaerobic
    "extreme": 1.00    # 100% anaerobic
}

# Storage structure for results
exertion_results = {
    ex_name: {
        "lactate": [],
        "lactic_acid": [],
        "lactate_total": [],
        "NH4": [],
        "pH": []
    }
    for ex_name in EXERTION_LEVELS.keys()
}

N_MC = 300 # number of iterations
t_end = 240.0
time = np.arange(0, t_end + 1e-9, 10.0)

T_current = 30.0
Q10_fixed = 2.5

# Run exertion simulation
results = simulate_exertion_levels(acid_factor_opt, k_excr_NH4_opt, k_CA_opt)

# Fill the plotting structure
for ex_name in EXERTION_LEVELS.keys():
    exertion_results[ex_name]["lactate"]        = results[ex_name]["lactate"]
    exertion_results[ex_name]["lactic_acid"]    = results[ex_name]["lactic_acid"]
    exertion_results[ex_name]["lactate_total"]  = [
        L + LA for L, LA in zip(results[ex_name]["lactate"],
                                results[ex_name]["lactic_acid"])
    ]
    exertion_results[ex_name]["NH4"]            = results[ex_name]["NH4"]
    exertion_results[ex_name]["pH"]             = results[ex_name]["pH"]

# =========================
# 20. Plotting the results
# =========================
colors = {
    "rest":    "black",
    "mild":    "green",
    "burst":   "blue",
    "extreme": "red"
}

fig, ax1 = plt.subplots(figsize=(10,6))

# -----------------------------
# LEFT AXIS — LACTATE
# -----------------------------
for ex_name in EXERTION_LEVELS.keys():

    lac_arr = np.array(exertion_results[ex_name]["lactate_total"])
    if lac_arr.size == 0:
        continue

    lac_mean = lac_arr.mean(axis=0)
    lac_sd   = lac_arr.std(axis=0)

    ax1.plot(time, lac_mean,
             color=colors[ex_name],
             linewidth=2,
             label=f"Lactate total — {ex_name}")

    ax1.fill_between(time,
                     lac_mean - lac_sd,
                     lac_mean + lac_sd,
                     color=colors[ex_name],
                     alpha=0.15)

ax1.set_xlabel("Time (min)")
ax1.set_ylabel("Lactate total (µmol/g)", color="black")
ax1.tick_params(axis='y', labelcolor="black")

# -----------------------------
# RIGHT AXIS — pH
# -----------------------------
ax2 = ax1.twinx()

for ex_name in EXERTION_LEVELS.keys():

    pH_arr = np.array(exertion_results[ex_name]["pH"])
    if pH_arr.size == 0:
        continue

    pH_mean = pH_arr.mean(axis=0)
    pH_sd   = pH_arr.std(axis=0)

    ax2.plot(time, pH_mean,
             color=colors[ex_name],
             linestyle="--",
             linewidth=2,
             label=f"pH — {ex_name}")

    ax2.fill_between(time,
                     pH_mean - pH_sd,
                     pH_mean + pH_sd,
                     color=colors[ex_name],
                     alpha=0.10)

ax2.set_ylabel("Blood pH", color="black")
ax2.tick_params(axis='y', labelcolor="black")

# -----------------------------
# COMBINED LEGEND
# -----------------------------
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="upper left", fontsize=10)

plt.title("Lactate total (solid) and pH (dashed) Across Exertion Levels")
plt.grid(True)
plt.tight_layout()
plt.show()
