import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

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

for df in [metabolite_table, rxn_df, enzyme_df]:
    df.columns = df.columns.str.strip().str.lower()

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
LAC_pKa    = 3.86

# Approximate blood water fraction (e.g. 80% of blood mass is water)
blood_water_fraction = 0.80
blood_water_mass = blood_mass * blood_water_fraction  # g water

# =========================
# 2. Normalise the data according to temperature and Q10
# =========================

T_ref = 30.0   # reference temperature (°C)
Q10   = 2.5    # Q10
Q10_fixed = 2.5

def metabolic_rate_umolO2_min(body_mass_g, T_C, Q10_local=Q10):
    """
    Returns whole-animal metabolic rate in µmol O2/min
    using alligator SMR at 30°C and Q10 scaling.
    """
    body_mass_kg = body_mass_g / 1000.0

    # SMR at 30°C (mL O2/min), a * M^0.83
    a = 0.20  # mL O2·kg^-1·min^-1 at 30°C
    SMR_mlO2_min_30 = a * (body_mass_kg ** 0.83)

    # Q10 scaling
    SMR_mlO2_min_T = SMR_mlO2_min_30 * (Q10_local ** ((T_C - T_ref) / 10.0))

    # Convert to µmol O2/min
    return SMR_mlO2_min_T * 44.6

T_current = 30.0  # °C

MR_umolO2_min = metabolic_rate_umolO2_min(body_mass, T_current)

# =========================
# 3. build metabolite-enzyme data structure
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
# 4. bui.d the organ model
# =========================
def build_organ_model(organ_name):
    m_sub = metabolite_table[metabolite_table["compartment"] == organ_name]
    e_sub = enzyme_df[enzyme_df["compartment"] == organ_name]
    if e_sub.empty:
        return None

    # -----------------------------
    # 1. Build reaction list
    # -----------------------------
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

    # -----------------------------
    # 2. Metabolite universe
    # -----------------------------
    mets = set()
    for r in rxns:
        mets.update(r["subs"])
        mets.update(r["prods"])

    for acid_met in [H_NAME, HCO3_NAME, CO2_NAME, LAC_NAME, LACH_NAME, NH4_NAME]:
        mets.add(acid_met)

    metabolites = sorted(mets)
    met_idx = {m: i for i, m in enumerate(metabolites)}

    # -----------------------------
    # Build initial concentrations using BLOOD as baseline
    # -----------------------------
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
        raise RuntimeError("Use sampled organ model with k_f/k_r, not the base template.")

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
# 5. Now sample the organ models
# =========================
def sample_organ_model(base_org, alpha=0.1, rng=None): # alpha is the reversibility of the enzymatic reactions
    # alpha is meant to be tuned - but for the study 0.1/ 10% reversibility was used.
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
    # ODE SYSTEM (pH-aware, CO2/HCO3 equilibrium)
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

            HCO3_conc = max(y[i_HCO3], 1e-12)
            CO2_conc  = max(y[i_CO2], 1e-12)

            # H_conc is in µmol/g
            H_umol_per_g = max(y[i_H], 1e-12)

            # convert to mol/L
            H_mol_per_L = (H_umol_per_g * 1e-6) * 1060.0   # 1060 g blood per L

            # compute pH
            pH = -np.log10(max(H_mol_per_L, 1e-20))
            pK_prime = -0.0817 * pH + 6.7818

            C_T = HCO3_conc + CO2_conc

            if C_T > 0.0 and np.isfinite(pH) and np.isfinite(pK_prime):
                ratio = 10.0 ** (pH - pK_prime)
                CO2_eq  = C_T / (1.0 + ratio)
                HCO3_eq = C_T - CO2_eq

                CO2_eq  = max(CO2_eq, 1e-12)
                HCO3_eq = max(HCO3_eq, 1e-12)

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

def build_sampled_organs(rng=None):
    if rng is None:
        rng = np.random.default_rng()
    organs_mc = {}
    for name, base_org in organs.items():
        organs_mc[name] = sample_organ_model(base_org, rng=rng)
    return organs_mc

# =========================
# 6. Now rebuild the organ models
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
# 7. Now for metabolic rate
# =========================
# All organs except blood are active; blood is the shared compartment
ACTIVE_ORGANS = [o for o in organs.keys() if o != "blood"]

# Baseline metabolic acid load into blood (µmol/(g_blood·min))
baseline_acid_load = 0.0  # set >0 if you want a constant acid load

# see the GitHub for an explanation of this
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
# 8. Global model with the metabolic rates etc
# =========================
# the acid factor of one assumes the standard rate of acid prod, gamma presumes the rate of lactate->lactici acid and the dissociation curve
# k_CA assumes the activity of carbonic anhydrase
def build_global_model_with_orgs(
        active_orgs,
        organs_mc,
        lac_rate_per_g=0.0,
        nh4_rate_per_g=0.0,
        baseline_acid_load=0.0,
        k_excr_NH4=0.0,
        k_CA=0.0,
        acid_factor=1.0,
        gamma=1.7
    ):
    """
    active_orgs: list of organ names (excluding blood)
    organs_mc: sampled organ models
    lac_rate_per_g, nh4_rate_per_g: whole-animal metabolic loads
    baseline_acid_load: constant H+ load into blood
    """

    # ---------------------------------------------------------
    # 1. GLOBAL METABOLITE INDEX
    # ---------------------------------------------------------
    org_names = active_orgs
    n_org = len(org_names)

    all_mets = sorted({m for o in org_names for m in organs_mc[o]["metabolites"]})
    met_idx_global = {m: i for i, m in enumerate(all_mets)}
    n_m = len(all_mets)

    # ---------------------------------------------------------
    # 2. INITIAL CONDITIONS BUILDER
    # ---------------------------------------------------------
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
        y0_parts.append(blood0)

        return np.concatenate(y0_parts)

    # ---------------------------------------------------------
    # 3. GLOBAL ODE SYSTEM
    # ---------------------------------------------------------
    def global_odes(t, y):
        
        # Clamp state
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
        # BASELINE ACID LOAD (OPTIONAL)
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

        if nh4_idx is not None and HCO3_idx is not None:
            NH4_blood  = blood[nh4_idx]   # µmol/g
            # total NH4+ excretion flux (µmol/min)
            J_excr_NH4 = k_excr_NH4 * NH4_blood * blood_mass
            # convert to per g (µmol/g/min) for blood compartment
            excr_rate_per_g = J_excr_NH4 / blood_mass
            # NH4+ sink
            d_blood[nh4_idx]  -= excr_rate_per_g
            # HCO3- sink (equimolar)
            d_blood[HCO3_idx] -= excr_rate_per_g

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
# 9. reset the blood pH equilibrium to enable the clincial relevance and real time updates to pH
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
# 10. modelling the NH4 at rest state and the hyperparameterisation
# =========================
# NH4HCO3 renal excretion parameters
NH4_EXCR_TOTAL_REST = 6.1   # µmol/min for 1.5 kg animal (scaled from 4.3 µmol/min at 1.05 kg)

NH4_REST_CONC = 2.0         # µmol/g, choose a plausible resting NH4+ concentration
blood_mass_g   = blood_mass # you already have this

# clearance constant k_excr (min^-1)
k_excr_NH4 = NH4_EXCR_TOTAL_REST / (NH4_REST_CONC * blood_mass_g)

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
        anaer_frac=1.00, # determined the rate of anaerobic... 0.05=5%, 0.1=10% etc etc
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
print("Optimized parameters:", res.x)

best_theta = res.x
acid_factor_opt = best_theta[0]
k_excr_NH4_opt  = best_theta[1]
k_CA_opt        = best_theta[2]

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

# =========================
# 11. time for the Monte Carlo modelling
# =========================
N_MC = 300
t_end = 240.0
time = np.arange(0, t_end + 1e-9, 10.0)

T_current = 30.0
Q10_fixed = 2.5

for ex_name, anaer_frac in EXERTION_LEVELS.items():

    print(f"\n=== Exertion level: {ex_name} (anaerobic fraction = {anaer_frac}) ===")

    for mc in range(N_MC):

        # ------------------------------------------------------------
        # 1. metabolic rate → lactate & NH4 load
        # ------------------------------------------------------------
        MR_umolO2_min = metabolic_rate_umolO2_min(body_mass, T_current, Q10_fixed)

        # exertion‑regulated anaerobic metabolism
        anaerobic_rate = MR_umolO2_min * anaer_frac

        # lactate production from anaerobic ATP generation
        lac_total_rate = (2.0 / 6.0) * anaerobic_rate

        # NH4 production proportional to lactate
        nh4_total_rate = 0.25 * lac_total_rate

        lac_rate_per_g = lac_total_rate / blood_mass
        nh4_rate_per_g = nh4_total_rate / blood_mass
        
        gamma = 1.7
        effective_acid_load = acid_factor_opt * (anaer_frac ** gamma)

        # ------------------------------------------------------------
        # 2. sample organs
        # ------------------------------------------------------------
        rng = np.random.default_rng(1000 + mc)
        organs_mc = {
            name: sample_organ_model(base_org, rng=rng)
            for name, base_org in organs.items()
            if name != "blood"
        }

        # ------------------------------------------------------------
        # 3. build global model (NO BOLUS)
        # ------------------------------------------------------------
        global_odes, build_C0, met_idx_global = build_global_model_with_orgs(
            ACTIVE_ORGANS,
            organs_mc,
            lac_rate_per_g=lac_rate_per_g,
            nh4_rate_per_g=nh4_rate_per_g,
            baseline_acid_load=effective_acid_load,
            k_excr_NH4=k_excr_NH4_opt,
            k_CA=k_CA_opt
        )


        # ------------------------------------------------------------
        # 4. solve ODE system
        # ------------------------------------------------------------
        y0 = build_C0()

        # Set initial total lactate (µmol/g) and pH
        L_tot0 = 6.095     # choose your starting blood lactate
        pH0    = 7.08

        y0 = reset_blood_lactate_equilibrium(y0, met_idx_global, L_tot0, pH0)

        sol = solve_ivp(global_odes, (0, t_end), y0, method="BDF", t_eval=time)

        # ------------------------------------------------------------
        # 5. extract blood block
        # ------------------------------------------------------------
        n_m = len(met_idx_global)
        blood_block = sol.y[-n_m:, :]
        
        H_idx   = met_idx_global[H_NAME]
        lac_idx = met_idx_global[LAC_NAME]
        lach_idx = met_idx_global[LACH_NAME]
        LACH_trace = blood_block[lach_idx, :]
        exertion_results[ex_name]["lactic_acid"].append(LACH_trace)
        nh4_idx = met_idx_global[NH4_NAME]

        H_trace = blood_block[H_idx, :]
        H_umol_per_g = np.maximum(H_trace, 1e-12)
        H_mol_per_L  = H_umol_per_g * 1e-6 * 1060.0  # 1060 g/L

        pH_trace = -np.log10(np.maximum(H_mol_per_L, 1e-20))

        LAC_trace = blood_block[lac_idx, :]
        nh4_trace     = blood_block[nh4_idx, :]

        blood_block = sol.y[-n_m:, :]     # all time points
        blood_final = blood_block[:, -1]  # final time point only

        # ------------------------------------------------------------
        # 6. store results
        # ------------------------------------------------------------
        Ltot_trace = LAC_trace + LACH_trace
        exertion_results[ex_name]["lactate_total"].append(Ltot_trace)
        exertion_results[ex_name]["lactate"].append(LAC_trace)
        exertion_results[ex_name]["lactic_acid"].append(LACH_trace)
        exertion_results[ex_name]["NH4"].append(nh4_trace)
        exertion_results[ex_name]["pH"].append(pH_trace)
        
# =========================
# 12. plotting of the Monte Carlo simulation
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