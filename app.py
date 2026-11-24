from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Deteksi Akun Palsu Instagram",
    page_icon=":material/verified_user:",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "model_terbaik.pkl"
THRESHOLD = 0.5


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


def digit_ratio(text):
    return sum(c.isdigit() for c in text) / len(text) if text else 0.0


YES_NO = ["Tidak", "Ya"]

CONTOH = {
    "Akun asli": {
        "username": "rina.sari", "fullname": "Rina Sari", "bio": "Pecinta kopi | Jakarta",
        "profile_pic": "Ya", "external_url": "Tidak", "private": "Tidak",
        "posts": 120, "followers": 850, "follows": 400,
    },
    "Akun palsu": {
        "username": "user84736291", "fullname": "", "bio": "",
        "profile_pic": "Tidak", "external_url": "Tidak", "private": "Tidak",
        "posts": 0, "followers": 12, "follows": 980,
    },
}

DEFAULTS = {
    "username": "", "fullname": "", "bio": "",
    "profile_pic": "Ya", "external_url": "Tidak", "private": "Tidak",
    "posts": 0, "followers": 0, "follows": 0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def isi_contoh(nama):
    for k, v in CONTOH[nama].items():
        st.session_state[k] = v


def reset_form():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2.5rem; }
    h1 { letter-spacing: -0.02em; text-wrap: balance; }
    h3 { font-size: 1.125rem !important; padding-top: 0.25rem !important; }

    .hasil {
        border: 1px solid var(--hasil-garis);
        background: var(--hasil-latar);
        border-radius: 0.75rem;
        padding: 1.5rem 1.5rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgb(28 35 33 / 0.06), 0 8px 24px -12px rgb(28 35 33 / 0.18);
    }
    .hasil.palsu { --hasil-garis: #E0B4A8; --hasil-latar: #FBEFEB; --hasil-warna: #9A3220; }
    .hasil.asli  { --hasil-garis: #A9CBC5; --hasil-latar: #EBF4F2; --hasil-warna: #1B5450; }
    .hasil.kosong { --hasil-garis: var(--st-border-color, #D6D2C6); --hasil-latar: #F1EFE8; --hasil-warna: #4A524F; }

    .hasil .label { font-size: 0.875rem; color: #4A524F; margin: 0; }
    .hasil .angka {
        font-size: 3rem; font-weight: 700; line-height: 1.1;
        letter-spacing: -0.03em; color: var(--hasil-warna);
        font-variant-numeric: tabular-nums; margin: 0.25rem 0 0.75rem;
    }
    .hasil .vonis { font-size: 1.125rem; font-weight: 600; color: var(--hasil-warna); margin: 0 0 1rem; }
    .hasil .catatan { font-size: 0.8125rem; color: #4A524F; margin: 1rem 0 0; line-height: 1.5; max-width: 65ch; }

    .bar { position: relative; height: 0.625rem; border-radius: 999px; background: #DAD6CA; }
    .bar > .isi { position: absolute; inset: 0 auto 0 0; border-radius: 999px; background: var(--hasil-warna); transition: width 0.4s cubic-bezier(0.16, 1, 0.3, 1); }
    .bar > .batas { position: absolute; top: -0.25rem; bottom: -0.25rem; left: 50%; width: 2px; background: #1C2321; opacity: 0.55; }
    .skala { display: flex; justify-content: space-between; font-size: 0.75rem; color: #4A524F; margin-top: 0.375rem; font-variant-numeric: tabular-nums; }

    @media (prefers-reduced-motion: reduce) { .bar > .isi { transition: none; } }

    ::selection { background: #1F5F5B; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Deteksi Akun Palsu Instagram")
st.write(
    "Masukkan data profil akun yang ingin dicek. Hasil diperbarui otomatis setiap data berubah."
)

form_col, hasil_col = st.columns([3, 2], gap="large")

with form_col:
    with st.container(horizontal=True):
        st.button("Contoh akun asli", icon=":material/person_check:", on_click=isi_contoh, args=("Akun asli",))
        st.button("Contoh akun palsu", icon=":material/person_alert:", on_click=isi_contoh, args=("Akun palsu",))
        st.button("Kosongkan", icon=":material/restart_alt:", type="tertiary", on_click=reset_form)

    with st.container(border=True):
        st.subheader("Identitas")
        a, b = st.columns(2)
        a.text_input("Username", key="username", placeholder="rina.sari")
        b.text_input("Nama lengkap", key="fullname", placeholder="Rina Sari")
        st.text_area("Bio", key="bio", max_chars=150, height=88, placeholder="Pecinta kopi | Jakarta")

    with st.container(border=True):
        st.subheader("Aktivitas")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Post", min_value=0, max_value=10000, step=1, key="posts")
        c2.number_input("Follower", min_value=0, max_value=20000000, step=1, key="followers")
        c3.number_input("Following", min_value=0, max_value=10000, step=1, key="follows")

    with st.container(border=True):
        st.subheader("Profil")
        d1, d2, d3 = st.columns(3)
        d1.radio("Punya foto profil?", YES_NO, horizontal=True, key="profile_pic")
        d2.radio("Ada URL eksternal?", YES_NO, horizontal=True, key="external_url")
        d3.radio("Akun privat?", YES_NO, horizontal=True, key="private")

u = st.session_state.username.strip()
n = st.session_state.fullname.strip()
bio = st.session_state.bio

# Fitur turunan dihitung otomatis dari input, sama dengan cara data latih dibuat
features = {
    "profile pic": 1 if st.session_state.profile_pic == "Ya" else 0,
    "nums/length username": digit_ratio(u),
    "fullname words": len(n.split()),
    "nums/length fullname": digit_ratio(n),
    "name==username": int(bool(n) and n.lower() == u.lower()),
    "description length": len(bio),
    "external URL": 1 if st.session_state.external_url == "Ya" else 0,
    "private": 1 if st.session_state.private == "Ya" else 0,
    "#posts": int(st.session_state.posts),
    "#followers": int(st.session_state.followers),
    "#follows": int(st.session_state.follows),
}

with hasil_col:
    if not u:
        st.markdown(
            """
            <div class="hasil kosong" role="status">
              <p class="label">Probabilitas akun palsu</p>
              <p class="angka">&ndash;</p>
              <p class="vonis">Menunggu data</p>
              <p class="catatan">Isi username di kolom kiri, atau pakai salah satu tombol contoh, untuk melihat hasil.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        proba = float(model.predict_proba(pd.DataFrame([features]))[0, 1])
        palsu = proba >= THRESHOLD
        kelas = "palsu" if palsu else "asli"
        vonis = "Kemungkinan akun palsu" if palsu else "Kemungkinan akun asli"
        persen = f"{proba:.1%}"
        st.markdown(
            f"""
            <div class="hasil {kelas}" role="status">
              <p class="label">Probabilitas akun palsu</p>
              <p class="angka">{persen}</p>
              <p class="vonis">{vonis}</p>
              <div class="bar" role="img" aria-label="Probabilitas akun palsu {persen}, batas keputusan 50%">
                <div class="isi" style="width:{proba * 100:.1f}%"></div>
                <div class="batas"></div>
              </div>
              <div class="skala"><span>0%</span><span>batas 50%</span><span>100%</span></div>
              <p class="catatan">Perkiraan dari pola data latih, bukan keputusan final. Ambang 50% dipakai sebagai default.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.subheader("Fitur yang dihitung")
        rasio = features["#followers"] / features["#follows"] if features["#follows"] else None
        tabel = pd.DataFrame(
            {
                "Fitur": [
                    "Rasio angka di username",
                    "Kata di nama lengkap",
                    "Panjang bio",
                    "Nama sama dengan username",
                    "Follower per following",
                ],
                "Nilai": [
                    f"{features['nums/length username']:.2f}",
                    str(features["fullname words"]),
                    f"{features['description length']} karakter",
                    "Ya" if features["name==username"] else "Tidak",
                    f"{rasio:.2f}" if rasio is not None else "–",
                ],
            }
        )
        st.dataframe(tabel, hide_index=True, width="stretch")
        st.caption("Dihitung otomatis dari input, dengan cara yang sama seperti saat data latih dibuat.")
