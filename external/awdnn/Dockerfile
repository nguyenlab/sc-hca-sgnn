# AWDNN reentrancy detector — reproducible build for TOSEM R2 baseline.
#
# Upstream repo: https://github.com/sbanning/AWDNN
# Pinned commit: b1c6dc3be13043dd89b799d597452bc288d54e8f
#
# Why not use upstream's AWDNNA.py directly?
#   - It trains + tests in one pass via an internal 80/20 split; no save/load
#     of Word2Vec embeddings or Keras weights; `save_weights`/`load_weights`
#     are commented out (widennet_att.py:91, 94).
#   - `parameter_parser()` is called at module-import time in
#     config/models/widennet_att.py:28 and config/attention/Custom_Attention.py:10,
#     which blocks programmatic use.
#
# Our driver (external/awdnn/awdnn_driver.py) re-implements the same model
# verbatim (Normalization + Dense(192) → Dense(320) → CustomAttention(units=8)
# → Concat with wide → Flatten → Dense(2, softmax), binary_crossentropy over
# 2-class softmax) with four explicit modes: reproduce, train, infer, cross.
#
# Paper/code/log hyperparameter discrepancies are documented in README.md.

FROM tensorflow/tensorflow:2.9.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pin deps. gensim 4.3.2 exposes the KeyedVectors save/load API our driver
# uses. Protobuf < 4 is required by tf 2.9.0 (tf 2.9 vendors protobuf 3.x
# bindings; pip pulls protobuf 4+ for other packages and breaks tf if we
# don't cap it). scikit-learn + pandas + numpy pinned to versions known to
# work with tf 2.9's numpy 1.22.
RUN pip install --upgrade pip==23.3.2 \
    && pip install \
        "gensim==4.3.2" \
        "scikit-learn==1.3.2" \
        "pandas==2.0.3" \
        "numpy==1.22.4" \
        "protobuf<4"

# Clone upstream repo for its bundled contracts_re.txt (the "Qian et al. 2023
# reentrancy corpus" referenced in AWDNN's paper § 4.1, in reality a curated
# 393-fragment subset: 200 clean + 193 reentrancy).
ARG AWDNN_COMMIT=b1c6dc3be13043dd89b799d597452bc288d54e8f
WORKDIR /opt
RUN git clone https://github.com/sbanning/AWDNN.git awdnn-src \
    && cd awdnn-src \
    && git checkout ${AWDNN_COMMIT}

# Copy our driver + preprocessor.
COPY awdnn_driver.py /opt/awdnn_driver.py
COPY preprocess_sol_to_fragments.py /opt/preprocess_sol_to_fragments.py

# Sanity: imports resolve, TF recognizes GPU if exposed.
RUN python -c "import tensorflow as tf; import gensim; import sklearn; \
               print('tf', tf.__version__, 'gensim', gensim.__version__, \
                     'sklearn', sklearn.__version__)"

WORKDIR /work

# Default entrypoint is python + our driver; pass --mode and flags at `docker run`.
ENTRYPOINT ["python", "/opt/awdnn_driver.py"]
