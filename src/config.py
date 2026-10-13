"""
ANNITIA Challenge 2026 — Configuration centrale.
MIT License.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TRAIN_CSV      = DATA / "train.csv"
TEST_CSV       = DATA / "test.csv"
SAMPLE_SUB_CSV = DATA / "hello_world_submission.csv"

# ---------------------------------------------------------------------------
# Colonnes identifiantes et cibles — à NE JAMAIS utiliser en features
# ---------------------------------------------------------------------------
ID_COLS       = ["patient_id_anon", "trustii_id"]
TARGET_COLS   = [
    "evenements_hepatiques_majeurs",
    "evenements_hepatiques_age_occur",
    "death",
    "death_age_occur",
]

# Colonnes "statiques" (une seule valeur par patient)
STATIC_COLS = [
    "gender", "T2DM", "Hypertension", "Dyslipidaemia",
    "bariatric_surgery", "bariatric_surgery_age",
]

# ---------------------------------------------------------------------------
# Biomarqueurs longitudinaux (préfixes des colonnes _v1 .. _vK)
# ---------------------------------------------------------------------------
BIOMARKER_PREFIXES = [
    "BMI",                          # anthropométrie
    "alt", "ast", "bilirubin",      # tests hépatiques
    "chol", "ggt", "gluc_fast",
    "plt", "triglyc",               # plaquettes + lipides
    "aixp_aix_result_BM_3",         # Aixplorer (élastographie ultrasonore)
    "fibrotest_BM_2",               # FibroTest (biologique)
    "fibs_stiffness_med_BM_1",      # FibroScan / VCTE (kPa)
]

# Seuils cliniques pour les features "fraction of visits above threshold"
CLINICAL_THRESHOLDS = {
    "fib4": 2.67,                   # FIB-4 élevé
    "apri": 1.0,                    # APRI élevé
    "fibs_stiffness_med_BM_1": 12.0,  # FibroScan cirrhose probable (kPa)
    "fibrotest_BM_2": 0.58,         # FibroTest fibrose avancée
    "aixp_aix_result_BM_3": 10.0,   # seuil Aixplorer (à ajuster)
    "alt": 40.0, "ast": 40.0, "ggt": 55.0,
    "gluc_fast": 7.0,               # diabète (mmol/L)
    "BMI": 30.0,                    # obésité
    "nfs": 0.676,                   # NFS élevé (fibrose avancée)
    "ggt_plt": 0.5,                 # GGT/PLT ratio
}

# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5

# Poids officiels du challenge
W_HEPATIC = 0.7
W_DEATH   = 0.3
