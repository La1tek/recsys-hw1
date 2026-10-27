#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"
python_bin="${PYTHON:-python3}"
venv_python=".venv/bin/python"
usage="Использование: bash $0 [--skip-install]"

if (( $# > 1 )); then
    echo "$usage" >&2
    exit 2
fi
case "${1:-}" in
    ""|--skip-install) ;;
    -h|--help)
        echo "$usage"
        echo "Выбор Python: PYTHON=python3.11 bash $0"
        exit 0
        ;;
    *) echo "Неизвестный аргумент: $1. $usage" >&2; exit 2 ;;
esac

# Окружение: используем существующее или создаём новое.
if [[ -x "$venv_python" ]]; then
    python_bin="$venv_python"
elif [[ "${1:-}" == --skip-install ]]; then
    echo "Нет окружения .venv. Сначала запустите bash $0." >&2
    exit 1
fi
"$python_bin" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else "Нужен Python 3.10-3.12")'
if [[ ! -x "$venv_python" ]]; then
    "$python_bin" -m venv .venv
fi

# Зависимости можно пропустить при повторном запуске.
if [[ "${1:-}" != --skip-install ]]; then
    "$venv_python" -m pip install -r requirements.txt
fi

# Выполняем ноутбук в этом же окружении.
"$venv_python" -m ipykernel install --sys-prefix --name recsys-hw1
.venv/bin/jupyter-execute \
    --kernel_name=recsys-hw1 --timeout=1800 --inplace experiments.ipynb
echo "Готово: эксперименты сохранены в experiments.ipynb, графики и таблицы - в artifacts/."
