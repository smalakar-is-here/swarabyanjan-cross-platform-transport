# SOURCE CODE RECONCILIATION

## Finding

Standalone source files found in the byte-inspected artifact universe:

- `.py`: **0**
- `.sh`: **0**
- `.bat`: **0**
- `.ps1`: **0**

Therefore, no independent standalone source tree is supported by the inspected bytes. The canonical implementation is primarily notebook-embedded.

## Canonical notebook imports/path indicators

| Stage | Notebook | Parsed imports | Absolute paths detected |
| --- | --- | --- | --- |
| 01 | 01-dataset-acquisition-and-freeze.ipynb | datetime;hashlib;huggingface_hub;json;os;pathlib;platform;requests;requests.adapters;shutil;sys;tarfile;time;tqdm.auto;traceback;urllib.parse;urllib3.util.retry;zipfile | /kaggle/working |
| 02 | 02-p0-integrity-provenance-leakage-audit.ipynb | collections;csv;datetime;hashlib;importlib.metadata;json;math;numpy;openpyxl;os;pandas;pathlib;platform;pyarrow.parquet;re;shutil;sqlite3;subprocess;sys;unicodedata;zipfile | /kaggle/input;/kaggle/input.;/kaggle/working |
| 03 | 03-primary-model-development.ipynb | collections;csv;datetime;huggingface_hub;importlib.metadata;importlib.util;interpret.glassbox;joblib;json;numpy;os;pandas;pathlib;pyarrow;pyarrow.parquet;scipy.special;shutil;sklearn.feature_extraction.text;sklearn.linear_model;sklearn.metrics;sklearn.model_selection;sklearn.pipeline;subprocess;sys;tempfile;textwrap;torch;torch.utils.data;transformers | /kaggle/input;/kaggle/working |
| 04 | 04-calibration.ipynb | datetime;hashlib;importlib.metadata;importlib.util;interpret.glassbox;joblib;json;numpy;os;pathlib;scipy.optimize;scipy.special;sklearn.metrics;statsmodels.api;subprocess;sys;tempfile;textwrap;torch;torch.utils.data;transformers;unicodedata | /kaggle/input;/kaggle/working |
| 05 | 05-external-validation-and-h3-vfinal.ipynb | collections;datetime;hashlib;numpy;pandas;pathlib;scipy.special;scipy.stats;sklearn.metrics;statsmodels.api;statsmodels.stats.multitest | /kaggle/input;/kaggle/working |
| 06 | 06-fixed-budget-reliability-evaluation-vfinal.ipynb | collections;datetime;hashlib;importlib.metadata;importlib.util;json;matplotlib.pyplot;numpy;os;pathlib;subprocess;sys;tempfile;textwrap | /kaggle/input;/kaggle/working |
| 07 | 07-h3-h4-bridge-exploratory-vfinal.ipynb | collections;datetime;hashlib;numpy;pandas;pathlib;scipy.stats | /kaggle/input;/kaggle/working |
| 08 | 08-manuscript-figures-and-tables-vfinal.ipynb | collections;datetime;hashlib;matplotlib;matplotlib.lines;matplotlib.patches;matplotlib.pyplot;numpy;pandas;pathlib | /kaggle/input;/kaggle/working;/kaggle/working/08_manuscript_outputs;/mnt/data;/mnt/data/08_manuscript_outputs |
| 09 | 09-source-publisher-cluster-sensitivity-vfinal.ipynb | collections;csv;datetime;numpy;pandas;pathlib;pyarrow.parquet;scipy.stats;sklearn.metrics;statsmodels.stats.multitest;urllib.parse | /kaggle/input;/kaggle/working;/mnt/data |

Absolute paths are recorded as provenance/environment dependencies. They were **not** rewritten.

Historical Library metadata contains additional repair notebooks and packages, but those metadata-only records are not treated as recovered standalone source.
