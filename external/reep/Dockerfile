# ReEP reentrancy detector — reproducible build for TOSEM R2 baseline comparison.
#
# ReEP requires Python 3.8 strict (bundled Manticore fork pins old typing semantics).
# Runtime deps at a glance:
#   - z3-solver 4.11.2 (pip wheel, NOT source-built Z3 4.5 as the README text claims)
#   - Manticore-derived symbolic executor (vendored under core/ + executor/)
#   - Slither CLI on PATH (ReEP.py:155 shells out to `slither ... --json`)
#   - solc-select for pragma-aware compiler switching
# The 8 "origin tools" referenced in the ReEP paper are pre-computed CSVs under
# Experimental_data/, NOT runtime dependencies.

FROM python:3.8-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build tooling for manticore-stack native deps (capstone, unicorn, pyelftools),
# plus git for cloning ReEP and curl for solc-select downloads.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        pkg-config \
        libssl-dev \
        libgmp-dev \
        libleveldb-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pinned ReEP commit (zexuwang1/ReEP HEAD as of Step 0 artifact discovery).
ARG REEP_COMMIT=7dcad71ba87a36cfc8959a29e6c7ebcf464ece58

WORKDIR /opt
RUN git clone https://github.com/zexuwang1/ReEP.git reep-src \
    && cd reep-src \
    && git checkout ${REEP_COMMIT}

WORKDIR /opt/reep-src/ReEP_tool

# ReEP's requirement.txt is the author's full `pip freeze` and includes packages
# that are never imported + some that aren't on PyPI (teether==0.1.10, the
# placeholder "some-package==0.1"). Filter those out before installing.
# Verified via grep: none of teether/some-package/octopus/leb128/multiaddr/
# varint/netaddr/base58/thrift/waitress are imported anywhere under ReEP_tool/.
# crytic-compile==0.1.13 is the era-matched version for the vendored Slither
# 0.7.1 egg installed below — keep it (don't filter it out).
RUN pip install --upgrade pip setuptools==65.5.1 wheel \
    && grep -vE '^(teether|some-package|octopus|leb128|multiaddr|varint|netaddr|base58|thrift|waitress)==' requirement.txt \
       > requirement.filtered.txt \
    && pip install -r requirement.filtered.txt

# Build ReEP's native executor modules. install.sh itself starts with
# `wrokon python38` (typo for `workon`, a virtualenvwrapper command we don't
# need inside Docker) — inline the rest.
#
# We install ONLY executor/validator (ReEP's Manticore-0.3.4 fork). We
# deliberately SKIP executor/searcher/setup.py install: despite sharing a
# directory with Manticore-side code, searcher's setup.py packages a stock
# Slither 0.8.3 that lacks ReEP's tsse/unrestricted_write detector — running
# it would shadow the vendored 0.7.1 egg installed below. Verified grep shows
# ReEP.py imports `from slither import Slither` (resolved by the 0.7.1 egg)
# and never references executor.searcher as a Python module.
RUN cd executor/validator && python setup.py install

# Upstream bugs in ReEP detector/manti_reen1.py that we patch:
#
# (1) Line 470: passes the RESULT of m.multi_tx_analysis1(...) (None) as the
#     second argument to func_timeout instead of the callable — raises
#     `TypeError: 'NoneType' object is not callable` on every sample, short-
#     circuiting CSV output. Wrap in lambda so a callable is actually passed.
#
# (2) Line ~506: `newName = os.getcwd() + '/' + filename + '_' + contractname`
#     concatenates an absolute input path under os.getcwd(), producing e.g.
#     `/work//work/<file>_<contract>`. The subsequent `mv` silently fails
#     (parent dir doesn't exist) and the `open(newName + '/.state_id')` then
#     raises FileNotFoundError, so no CSV row is written even when Manticore
#     DID find the bug. Fix by using os.path.basename(filename).
RUN sed -i 's|func_timeout(30, m.multi_tx_analysis1(initcode, contract_name=contractname, args=symbolic_value,tx_limit=1,tx_preconstrain=False,tx_send_ether=True))|func_timeout(30, lambda: m.multi_tx_analysis1(initcode, contract_name=contractname, args=symbolic_value,tx_limit=1,tx_preconstrain=False,tx_send_ether=True))|' \
        /opt/reep-src/ReEP_tool/detector/manti_reen1.py \
    && sed -i "s|fname = filename +'_'+ contractname|fname = os.path.basename(filename) +'_'+ contractname|g" \
        /opt/reep-src/ReEP_tool/detector/manti_reen1.py \
    && grep -q 'func_timeout(30, lambda:' /opt/reep-src/ReEP_tool/detector/manti_reen1.py \
    && grep -q 'os.path.basename(filename)' /opt/reep-src/ReEP_tool/detector/manti_reen1.py

# Slither: use ReEP's VENDORED fork, not pip slither-analyzer.
# ReEP.py:155 shells out to `slither ... --detect unrestricted-write-state,...`,
# which is a CUSTOM detector ReEP ships under analyzer/slither/detectors/tsse/.
# Pip's upstream Slither does not have this detector. The vendored fork is
# slither_analyzer 0.7.1 (prebuilt egg at analyzer/dist/), compatible with
# crytic-compile==0.1.13 pinned by ReEP.
# Pip cannot install .egg files and easy_install was removed from setuptools
# 58+, so install the egg manually: drop it on sys.path via easy-install.pth
# and synthesize the `slither` console-script from the egg's entry_points.txt
# (verified: slither = slither.__main__:main).
RUN SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')" \
    && cp analyzer/dist/slither_analyzer-0.7.1-py3.8.egg "$SITE_PACKAGES/" \
    && echo "./slither_analyzer-0.7.1-py3.8.egg" >> "$SITE_PACKAGES/easy-install.pth" \
    && printf '#!/usr/local/bin/python\nimport sys\nfrom slither.__main__ import main\nsys.exit(main())\n' > /usr/local/bin/slither \
    && chmod +x /usr/local/bin/slither

# solc-select 1.0.2 from ReEP's requirement.txt uses a deprecated endpoint that
# returns HTTP 403 when pre-fetching the version list. Override with a modern
# release that hits binaries.soliditylang.org directly.
RUN pip install --upgrade 'solc-select>=1.0.4'

# Pre-install solc versions commonly seen in SmartBugs + our dataset.
# (More can be added at runtime with `solc-select install X.Y.Z`.)
RUN solc-select install \
        0.4.24 0.4.25 0.4.26 \
        0.5.0 0.5.16 \
        0.6.0 0.6.12 \
        0.7.0 0.7.6 \
        0.8.0 0.8.17 \
    && solc-select use 0.4.25

# Sanity: confirm entry-point imports cleanly.
RUN python -c "from slither import Slither; import z3; print('z3 version:', z3.get_version_string())"

# Working dir for per-sample invocations; mount the dataset here at run time.
WORKDIR /work

# Usage: docker run --rm -v /path/to/contracts:/work reep \
#          -f reentrancy_bonus.sol -r -a -o /work/out.csv
ENTRYPOINT ["python", "/opt/reep-src/ReEP_tool/ReEP.py"]
