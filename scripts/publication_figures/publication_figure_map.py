"""Single source of truth for the journal-presentation figure set (2026-11-15).

published stem -> (generator output stem, subdirectory, generator, canonical Stage08 counterpart, frozen sources)
"""
PUB_DIR = "results/publication/journal_presentation_2026-11-15"
F5 = ["FIG05_H3_H4_BRIDGE_abs_delta_rd__vs__lift_deterioration_source.csv",
      "FIG05_H3_H4_BRIDGE_abs_delta_rd__vs__error_capture_deterioration_source.csv",
      "FIG05_H3_H4_BRIDGE_abs_delta_nll_gap__vs__lift_deterioration_source.csv",
      "FIG05_H3_H4_BRIDGE_abs_delta_nll_gap__vs__error_capture_deterioration_source.csv"]
FIGURES = {
    "mainfig_01_study_design_v3": ("FIG01_STUDY_DESIGN_v3", "main", "make_fig1_v3.py", "FIG01_STUDY_DESIGN",
                                   "see FIG01_v3_SOURCE_MANIFEST.json (upstream Stage01-Stage10 records)"),
    "mainfig_02_h1_h2_v2": ("FIG02_H1_H2_v2", "main", "make_main_figures.py:fig2", "FIG02_H1_H2",
                            ["FIG02_H1_H2_A_source.csv", "FIG02_H1_H2_B_C_source.csv", "FIG02_H1_H2_D_source.csv"]),
    "mainfig_03_h3_v2": ("FIG03_H3_CUE_TRANSPORT_v2", "main", "make_main_figures.py:fig3", "FIG03_H3_CUE_TRANSPORT",
                         ["FIG03_H3_CUE_TRANSPORT_A_B_source.csv"]),
    "mainfig_04_h4_v2": ("FIG04_H4_FIXED_BUDGET_v2", "main", "make_main_figures.py:fig4", "FIG04_H4_FIXED_BUDGET",
                         ["FIG04_H4_FIXED_BUDGET_primary_10pct_source.csv",
                          "FIG04_H4_FIXED_BUDGET_budget_sensitivity_source.csv"]),
    "mainfig_05_h3_h4_bridge_v2": ("FIG05_H3_H4_BRIDGE_v2", "main", "make_main_figures.py:fig5", "FIG05_H3_H4_BRIDGE",
                                   F5),
    "suppfig_S1_h1_h2_model_seed_v2": ("SUPP_S1_H1_MODEL_SEED_v2", "supplementary", "make_supp_figures.py:s1",
                                       "SUPP_H1_H2_MODEL_SEED", ["SUPP_H1_H2_MODEL_SEED_all_source.csv"]),
    "suppfig_S2_h3_sparsity_v2": ("SUPP_S2_H3_SPARSITY_v2", "supplementary", "make_supp_figures.py:s2",
                                  "SUPP_H3_SPARSITY", ["SUPP_H3_SPARSITY_support_source.csv",
                                                       "FIG03_H3_CUE_TRANSPORT_A_B_source.csv",
                                                       "SUPP_BAITBUSTER_H3_A_B_source.csv"]),
    "suppfig_S3a_h3p_curiosity_v2": ("SUPP_S3a_H3P_CURIOSITY_v2", "supplementary", "make_supp_figures.py:s3",
                                     "SUPP_H3P_curiosity_information_withholding", ["SUPP_H3P_SHARED_all_source.csv"]),
    "suppfig_S3b_h3p_direct_address_v2": ("SUPP_S3b_H3P_DIRECT_ADDRESS_v2", "supplementary", "make_supp_figures.py:s3",
                                          "SUPP_H3P_direct_address_imperative", ["SUPP_H3P_SHARED_all_source.csv"]),
    "suppfig_S3c_h3p_punctuation_v2": ("SUPP_S3c_H3P_PUNCTUATION_v2", "supplementary", "make_supp_figures.py:s3",
                                       "SUPP_H3P_punctuation_intensity", ["SUPP_H3P_SHARED_all_source.csv"]),
    "suppfig_S4_baitbuster_h3_v2": ("SUPP_S4_BAITBUSTER_H3_v2", "supplementary", "make_supp_figures.py:s4",
                                    "SUPP_BAITBUSTER_H3", ["SUPP_BAITBUSTER_H3_A_B_source.csv"]),
    "suppfig_S5_h4_05pct_v2": ("SUPP_S5_H4_05PCT_v2", "supplementary", "make_supp_figures.py:s_h4", "SUPP_H4_05PCT",
                               ["SUPP_H4_05PCT_A_C_source.csv"]),
    "suppfig_S6_h4_20pct_v2": ("SUPP_S6_H4_20PCT_v2", "supplementary", "make_supp_figures.py:s_h4", "SUPP_H4_20PCT",
                               ["SUPP_H4_20PCT_A_C_source.csv"]),
    "suppfig_S7_h4_model_seed_v2": ("SUPP_S7_H4_MODEL_SEED_v2", "supplementary", "make_supp_figures.py:s7",
                                    "SUPP_H4_MODEL_SEED", ["SUPP_H4_MODEL_SEED_all_source.csv"]),
    "suppfig_S8_risk_coverage_v2": ("SUPP_S8_RISK_COVERAGE_v2", "supplementary", "make_supp_figures.py:s8",
                                    "SUPP_RISK_COVERAGE", ["SUPP_RISK_COVERAGE_all_source.csv"]),
    "suppfig_S9_bridge_loo_v2": ("SUPP_S9_BRIDGE_LOO_v2", "supplementary", "make_supp_figures.py:s9", "SUPP_BRIDGE_LOO",
                                 ["SUPP_BRIDGE_LOO_all_source.csv"] + F5),
    "suppfig_S10_prevalence_structural_sensitivity_v2": ("SUPP_S10_SENSITIVITY_v2", "supplementary",
                                                         "make_supp_figures.py:s10", "SUPP_SENSITIVITY",
                                                         ["SUPP_SENSITIVITY_A_B_source.csv"]),
}
EXTS = ("pdf", "svg", "png")
WORDING_DIFFERENCES = [
    {"canonical": "results/primary/stage08/figures/main/FIG05_H3_H4_BRIDGE.*", "canonical_wording": "Youtube",
     "publication_figure": "mainfig_05_h3_h4_bridge_v2", "publication_wording": "YouTube"},
    {"canonical": "results/primary/stage08/figures/supplementary/SUPP_BAITBUSTER_H3.*",
     "canonical_wording": "independent-collection H3 replication",
     "publication_figure": "suppfig_S4_baitbuster_h3_v2",
     "publication_wording": "external corroboration from a related construct, not a construct-equivalent replication"},
]
