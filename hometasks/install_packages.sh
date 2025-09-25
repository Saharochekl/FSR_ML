#!/usr/bin/env bash
# install_packages.sh — macOS (M1/M2), совместим с bash 3.2
# Создаёт .venv, ставит нужные пакеты, проверяет импорты. Запускать: bash ./install_packages.sh

set -euo pipefail

# 0) Проверим, что это bash, а не zsh/sh
if [ -z "${BASH_VERSION:-}" ]; then
  echo "❌ Это не bash. Запусти так: bash ./install_packages.sh" >&2
  exit 1
fi

# 1) Python
if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ Не найден python3. Установи: brew install python" >&2
  exit 1
fi
PY="$(command -v python3)"
PYVER="$("$PY" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
echo "➡️  Python: $PY ($PYVER)"

# 2) venv
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "➡️  Создаю venv в $VENV_DIR"
  "$PY" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

# 3) pip инструменты
python -m pip install -U pip setuptools wheel >/dev/null

# 4) Маппинг: модуль -> pip-пакет (через case, без mapfile/assoc arrays)
mod2pkg () {
  case "$1" in
    numpy) echo numpy ;;
    scipy) echo scipy ;;
    statsmodels) echo statsmodels ;;
    sklearn) echo scikit-learn ;;
    seaborn) echo seaborn ;;
    matplotlib) echo matplotlib ;;
    pandas) echo pandas ;;
    *) echo "" ;;
  esac
}

# 5) Проверки импорта/символов
check_import () {
  python - "$1" <<'PY'
import importlib, sys
mod = sys.argv[1]
try:
    importlib.import_module(mod)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

check_symbols () {
  mod="$1"; csv_syms="$2"
  python - "$mod" "$csv_syms" <<'PY'
import importlib, sys
mod = importlib.import_module(sys.argv[1])
need = [s.strip() for s in sys.argv[2].split(",") if s.strip()]
missing = [s for s in need if not hasattr(mod, s)]
sys.exit(0 if not missing else 1)
PY
}

# 6) Список базовых модулей
PY_MODULES="scipy numpy statsmodels sklearn seaborn matplotlib pandas"
missing_pkgs=""

echo "➡️  Проверяю базовые импорты…"
for m in $PY_MODULES; do
  if check_import "$m"; then
    echo "   ✅ $m"
  else
    pkg="$(mod2pkg "$m")"
    if [ -n "$pkg" ]; then
      echo "   ❌ $m (добавляю к установке: $pkg)"
      missing_pkgs="$missing_pkgs $pkg"
    else
      echo "   ⚠️  $m: неизвестен соответствующий pip-пакет"
    fi
  fi
done

echo "➡️  Проверяю ключевые подмодули и символы…"
SYM_CHECKS="
scipy.interpolate:splrep,splev
scipy.stats:bws_test
statsmodels.api:OLS
statsmodels.stats.api:ttest_ind
sklearn.linear_model:LinearRegression,HuberRegressor,RANSACRegressor
sklearn.cluster:KMeans,DBSCAN,SpectralClustering,OPTICS,AgglomerativeClustering
sklearn.metrics:mean_squared_error,silhouette_score,adjusted_rand_score,accuracy_score,confusion_matrix
sklearn.svm:SVC
sklearn.decomposition:PCA
sklearn.cross_decomposition:CCA
sklearn.discriminant_analysis:LinearDiscriminantAnalysis
sklearn.preprocessing:StandardScaler,LabelEncoder
sklearn.model_selection:train_test_split
statsmodels.regression.quantile_regression:QuantReg
scipy.linalg:eigh
scipy.cluster.hierarchy:dendrogram,linkage
"

# Нормальные сообщения: если корневой модуль не установлен — пропускаем, не рисуем ложный "✅"
IFS=$'\n'
for item in $SYM_CHECKS; do
  mod="${item%%:*}"
  syms="${item#*:}"
  root="${mod%%.*}"
  pkg="$(mod2pkg "$root")"

  if ! check_import "$root"; then
    echo "   ⏭️  $mod: базовый модуль '$root' не установлен — сначала поставим его."
    [ -n "$pkg" ] && missing_pkgs="$missing_pkgs $pkg"
    continue
  fi

  if check_symbols "$mod" "$syms"; then
    echo "   ✅ $mod"
  else
    echo "   ❌ $mod: отсутствуют символы ($syms). Обновим $pkg."
    [ -n "$pkg" ] && missing_pkgs="$missing_pkgs $pkg"
  fi
done
unset IFS

# 7) Установим/обновим недостающее (уникализация без mapfile)
# shellcheck disable=SC2086
if [ -n "$missing_pkgs" ]; then
  pkgs="$(printf '%s\n' $missing_pkgs | sed '/^$/d' | sort -u | tr '\n' ' ')"
  echo "➡️  Ставлю/обновляю: $pkgs"
  # shellcheck disable=SC2086
  python -m pip install -U $pkgs
else
  echo "✅ Всё уже стоит."
fi

# 8) Проверка 'pc'
echo "➡️  Проверяю модуль 'pc'…"
if check_import "pc"; then
  echo "   ✅ pc"
else
  echo "   ⚠️  'pc' не найден: это либо локальный модуль, либо у него другое имя на PyPI."
fi

# 9) Финальная проба импортов
echo "➡️  Финальная проверка импортов…"
python - <<'PY'
import sys
try:
    from scipy.interpolate import splrep, splev
    import statsmodels, statsmodels.api as sm, statsmodels.stats.api as ssa
    import scipy, numpy as np, random, sklearn, seaborn as sns
    import scipy.stats as sps
    import matplotlib.pyplot as plt
    import math, csv, pandas as pd
    from scipy.stats import bws_test
    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
    from sklearn.metrics import mean_squared_error, silhouette_score, adjusted_rand_score, accuracy_score, confusion_matrix
    from statsmodels.regression.quantile_regression import QuantReg
    from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, OPTICS, AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.linalg import eigh as sp_eigh
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
    try:
        import pc
    except Exception:
        pass
except ModuleNotFoundError as e:
    print(f"❌ ModuleNotFoundError: {e}", file=sys.stderr); sys.exit(1)
except AttributeError as e:
    print(f"❌ AttributeError: {e}", file=sys.stderr); sys.exit(2)
print("✅ Импорты ок (кроме, возможно, 'pc').")
PY

echo "🎉 Готово."