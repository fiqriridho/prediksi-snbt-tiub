import streamlit as st
import numpy as np
import joblib
import google.generativeai as genai

# Set page configuration to wide layout
st.set_page_config(
    page_title="Prediksi & Konsultan AI SNBT UB",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# load model dan scaler
model = joblib.load('model_knn.pkl')
scaler = joblib.load('scaler.pkl')

# konfigurasi Gemini
genai.configure(
    api_key=st.secrets["GEMINI_API_KEY"]
)

gemini = genai.GenerativeModel(
    "gemini-2.5-flash"
)


st.title("Prediksi Peluang SNBT Teknik Informatika UB")
st.markdown("---")

# Split the page into two columns (50/50)
col_left, col_right = st.columns(2)

# ================= LEFT COLUMN: INPUT & RESULT =================
with col_left:
    st.header("Input & Prediksi Nilai")
    st.write("Masukkan nilai try out UTBK Anda:")
    st.caption("Catatan: Aplikasi ini memprediksi peluang kelolosan berdasarkan pola data try out sebelumnya, bukan merupakan jaminan kelolosan mutlak.")
    
    # Sub-columns for inputs to make it neat
    sub_col1, sub_col2 = st.columns(2)
    
    with sub_col1:
        PU = st.number_input("Penalaran Umum", 0, 1000, value=st.session_state.get("PU_val", 500))
        PPU = st.number_input("Pengetahuan dan Pemahaman Umum", 0, 1000, value=st.session_state.get("PPU_val", 500))
        PBM = st.number_input("Pemahaman Bacaan dan Menulis", 0, 1000, value=st.session_state.get("PBM_val", 500))
        PK = st.number_input("Pengetahuan Kuantitatif", 0, 1000, value=st.session_state.get("PK_val", 500))
    
    with sub_col2:
        LBI = st.number_input("Literasi Bahasa Inggris", 0, 1000, value=st.session_state.get("LBI_val", 500))
        LBI2 = st.number_input("Literasi Bahasa Indonesia", 0, 1000, value=st.session_state.get("LBI2_val", 500))
        PM = st.number_input("Penalaran Matematika", 0, 1000, value=st.session_state.get("PM_val", 500))
    
    # Store individual inputs in session state to preserve across chat reruns
    st.session_state["PU_val"] = PU
    st.session_state["PPU_val"] = PPU
    st.session_state["PBM_val"] = PBM
    st.session_state["PK_val"] = PK
    st.session_state["LBI_val"] = LBI
    st.session_state["LBI2_val"] = LBI2
    st.session_state["PM_val"] = PM

    if st.button("Prediksi Peluang Kelolosan", use_container_width=True, type="primary"):
        data = np.array([[PU, PPU, PBM, PK, LBI, LBI2, PM]])
        rata_rata = np.mean(data)
        
        # normalisasi
        data_scaled = scaler.transform(data)

        # prediksi
        prediction = model.predict(data_scaled)
        probability = model.predict_proba(data_scaled)
        
        st.session_state["hasil_prediksi"] = {
            "PU": PU,
            "PPU": PPU,
            "PBM": PBM,
            "PK": PK,
            "LBI": LBI,
            "LBI2": LBI2,
            "PM": PM,
            "rata_rata": rata_rata,
            "hasil": prediction[0],
            "keyakinan": float(np.max(probability) * 100)
        }

    # Render results if they exist in session_state
    if "hasil_prediksi" in st.session_state:
        res = st.session_state["hasil_prediksi"]
        
        st.markdown("---")
        st.subheader("📊 Hasil Analisis Prediksi")
        
        # Menggunakan kolom metrik untuk informasi yang lebih visual & ringkas
        m_col1, m_col2 = st.columns(2)
        m_col1.metric(label="Rata-rata Nilai", value=f"{res['rata_rata']:.2f}")
        m_col2.metric(label="Keyakinan Model", value=f"{res['keyakinan']:.2f}%")
        
        st.caption("Rata-rata nilai 620 digunakan sebagai indikator pembanding, sedangkan keputusan kelolosan utama ditentukan oleh model KNN.")

        if res["hasil"] == "BERPELUANG":
            st.success("🎉 Berpeluang Lolos SNBT Teknik Informatika UB")
        else:
            st.error("⚠️ Kurang Berpeluang Lolos SNBT Teknik Informatika UB")

        # Deteksi Kelemahan & Rekomendasi Belajar
        scores = {
            "Penalaran Umum": res["PU"],
            "Pengetahuan & Pemahaman Umum": res["PPU"],
            "Pemahaman Bacaan & Menulis": res["PBM"],
            "Pengetahuan Kuantitatif": res["PK"],
            "Literasi B. Inggris": res["LBI"],
            "Literasi B. Indonesia": res["LBI2"],
            "Penalaran Matematika": res["PM"]
        }
        
        subtes_terlemah = min(scores, key=scores.get)
        nilai_terendah = scores[subtes_terlemah]
        
        rekomendasi = {
            "Penalaran Umum": "Fokus pada latihan soal penalaran logis, analisis pola deret angka, serta penalaran spasial/gambar.",
            "Pengetahuan & Pemahaman Umum": "Perbanyak perbendaharaan kosakata (KBBI), sinonim/antonim, dan pahami hubungan makna kata dalam kalimat.",
            "Pemahaman Bacaan & Menulis": "Latih pemahaman ejaan yang disempurnakan (EYD), penggunaan tanda baca yang tepat, serta teknik menentukan ide pokok paragraf.",
            "Pengetahuan Kuantitatif": "Perkuat dasar aljabar, aritmatika sosial, geometri dasar, peluang, dan pemecahan masalah logika numerik.",
            "Literasi B. Inggris": "Tingkatkan kemampuan reading comprehension (memahami ide pokok bacaan Inggris) dan perkaya kosakata bahasa Inggris.",
            "Literasi B. Indonesia": "Fokus pada teknik membaca cepat (skimming/scanning), menarik kesimpulan bacaan, dan menganalisis paragraf argumentatif.",
            "Penalaran Matematika": "Perbanyak latihan mengubah soal cerita ke bentuk persamaan matematika, serta interpretasi grafik/tabel."
        }

        st.markdown("### 💡 Rekomendasi Belajar")
        st.info(f"Sub-tes dengan skor terendah Anda adalah **{subtes_terlemah}** ({nilai_terendah}).\n\n**Saran:** {rekomendasi[subtes_terlemah]}")

        # Section: Analisis Akademik Sederhana
        st.markdown("---")
        st.markdown("### 📊 Analisis Akademik")
        
        # Pencarian subtes terkuat & terlemah (dengan penanganan jika ada nilai yang sama)
        nilai_tertinggi = max(scores.values())
        subtes_terkuat_list = [k for k, v in scores.items() if v == nilai_tertinggi]
        subtes_terkuat_str = ", ".join(subtes_terkuat_list)

        subtes_terlemah_list = [k for k, v in scores.items() if v == nilai_terendah]
        subtes_terlemah_str = ", ".join(subtes_terlemah_list)
        
        # Metrik utama
        a_col1, a_col2 = st.columns(2)
        a_col1.metric("Rata-rata Nilai", f"{res['rata_rata']:.2f}")
        
        selisih = 620 - res['rata_rata']
        if selisih > 0:
            a_col2.metric("Selisih ke Target (620)", f"{selisih:.2f} poin", delta=f"-{selisih:.2f}")
        else:
            a_col2.metric("Selisih ke Target (620)", f"{abs(selisih):.2f} poin (Lolos)", delta=f"+{abs(selisih):.2f}")
            
        
        # Tampilan kekuatan & kelemahan
        st.success(f"💪 **Subtes Terkuat:** {subtes_terkuat_str} ({nilai_tertinggi})")
        st.warning(f"⚠️ **Subtes Terlemah:** {subtes_terlemah_str} ({nilai_terendah})")
        
        # Ringkasan analisis otomatis
        summary_lines = [
            f"Kekuatan utama berada pada **{subtes_terkuat_str}**.",
            f"Kelemahan utama berada pada **{subtes_terlemah_str}**."
        ]
        if selisih > 0:
            summary_lines.append(f"Rata-rata nilai masih berada **{selisih:.2f}** poin di bawah target 620.")
        else:
            summary_lines.append(f"Rata-rata nilai sudah berada **{abs(selisih):.2f}** poin di atas target 620.")

        if res["hasil"] == "BERPELUANG":
            if selisih > 0:
                st.info(f"💡 Meskipun rata-rata nilai belum mencapai 620, model KNN memprediksi Anda **BERPELUANG** lolos.")
            else:
                st.success(f"🎉 Rata-rata nilai Anda sudah melampaui target 620 sebesar {abs(selisih):.2f} poin.")
        else:
            if selisih > 0:
                st.error(f"🎯 Anda masih membutuhkan {selisih:.2f} poin untuk mencapai target 620.")
            else:
                st.warning(f"⚠️ Rata-rata nilai Anda sudah melampaui 620, namun model KNN memprediksi Anda **KURANG BERPELUANG** lolos.")

        summary_lines.append(f"Prioritas utama peningkatan adalah {subtes_terlemah_str} karena memiliki skor terendah.")        
        summary_markdown = "\n".join([f"- {line}" for line in summary_lines])
        st.info(f"📝 **Ringkasan Analisis:**\n\n{summary_markdown}")

    st.markdown("---")
    st.caption("Model dibangun menggunakan algoritma K-Nearest Neighbor (KNN) dengan Euclidean Distance.")

# ================= RIGHT COLUMN: AI CONSULTANT =================
with col_right:
    st.header("💬 Konsultan AI UTBK")
    st.write("Konsultasikan hasil nilai try out dan strategi belajar Anda dengan AI:")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    prompt = st.chat_input("Tanyakan sesuatu tentang hasil prediksi UTBK Anda...")

    if prompt:
        if "hasil_prediksi" not in st.session_state:
            st.warning("Silakan lakukan prediksi terlebih dahulu pada kolom kiri.")
        else:
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })

            with st.chat_message("user"):
                st.markdown(prompt)

            data = st.session_state["hasil_prediksi"]
            hasil_formatted = data['hasil'].replace('_', ' ').title()

            scores = {
                "Penalaran Umum": data["PU"],
                "Pengetahuan & Pemahaman Umum": data["PPU"],
                "Pemahaman Bacaan & Menulis": data["PBM"],
                "Pengetahuan Kuantitatif": data["PK"],
                "Literasi B. Inggris": data["LBI"],
                "Literasi B. Indonesia": data["LBI2"],
                "Penalaran Matematika": data["PM"]
            }

            nilai_max = max(scores.values())
            nilai_min = min(scores.values())

            subtes_terkuat = ", ".join(
                [k for k, v in scores.items() if v == nilai_max]
            )

            subtes_terlemah = ", ".join(
                [k for k, v in scores.items() if v == nilai_min]
            )

            selisih_target = 620 - data["rata_rata"]

            context = f"""
            Data peserta:

            PU = {data['PU']}
            PPU = {data['PPU']}
            PBM = {data['PBM']}
            PK = {data['PK']}
            LBI = {data['LBI']}
            LBI2 = {data['LBI2']}
            PM = {data['PM']}

            Hasil Prediksi:
            {hasil_formatted}

            Rata-rata Nilai:
            {data['rata_rata']:.2f}

            Subtes Terkuat:
            {subtes_terkuat}

            Subtes Terlemah:
            {subtes_terlemah}

            Selisih ke Target Rata-rata 620:
            {selisih_target:.2f}

            Kamu adalah konsultan UTBK.

            Aturan:
            - Analisis berdasarkan data di atas secara riil.
            - Fokus pada subtes terkuat dan terlemah.
            - JANGAN menyebut angka kenaikan poin tertentu secara spesifik (misal: "naikkan 10-20 poin").
            - JANGAN membuat prediksi numerik baru yang tidak ada pada data.
            - Gunakan hanya data nilai riil peserta yang diberikan.
            - Jangan memberi motivasi umum atau menyuruh membuat jadwal.
            - Jawab maksimal 100 kata.
            - Gunakan bullet point.
            - Maksimal 5 poin.
            """
            with st.spinner("Konsultan AI sedang menganalisis..."):
                response = gemini.generate_content(
                    context + "\n\nPertanyaan User:\n" + prompt
                )
                ai_reply = response.text

            with st.chat_message("assistant"):
                st.markdown(ai_reply)

            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_reply
            })
            st.rerun()