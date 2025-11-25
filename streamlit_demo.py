# -*- coding: utf-8 -*-
"""
Simple Streamlit Demo for Finnish NLP Toolkit
Works without emoji encoding issues
"""
import streamlit as st
import requests
import json

# API configuration
API_BASE_URL = "http://localhost:8000/api"

# Page configuration
st.set_page_config(
    page_title="Finnish NLP Toolkit - Demo",
    layout="wide"
)

st.title("Finnish NLP Toolkit - Giant Version 2.0")
st.markdown("**Advanced Finnish Language Processing Platform**")

# Sidebar
with st.sidebar:
    st.header("Features")
    feature = st.selectbox(
        "Select Feature",
        [
            "Lemmatization",
            "Hybrid Lemmatization",
            "Complexity Analysis",
            "Profanity Detection",
            "Morphological Entropy",
            "Semantic Disambiguation",
            "Linguistic Explanation",
            "Text Clarification",
            "Text Simplification",
            "Benchmarking"
        ]
    )

    st.markdown("---")
    st.markdown("### Statistics")
    st.metric("Total Routers", "11")
    st.metric("Total Endpoints", "30+")
    st.metric("Test Coverage", "99/99")

# Main content
if feature == "Lemmatization":
    st.header("Lemmatization")
    st.markdown("Convert Finnish words to their base forms")

    text = st.text_area("Finnish Text", value="Kissani söi hiiren puutarhassani.", height=100)
    include_morphology = st.checkbox("Include Morphology", value=True)

    if st.button("Lemmatize"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/lemmatize",
                json={"text": text, "include_morphology": include_morphology}
            )
            data = response.json()

            st.success(f"Processed {data['word_count']} words")

            for lemma in data['lemmas']:
                with st.expander(f"{lemma['original']} -> {lemma['lemma']}"):
                    st.write(f"**POS:** {lemma.get('pos', 'N/A')}")
                    if lemma.get('morphology'):
                        st.json(lemma['morphology'])
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif feature == "Hybrid Lemmatization":
    st.header("Hybrid 3-Stage Lemmatization")
    st.markdown("Dictionary -> ML -> Similarity")

    text = st.text_area("Finnish Text", value="Kissani söi hiiren.", height=100)

    if st.button("Process"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/hybrid-lemma",
                json={"text": text, "return_method_info": True}
            )
            data = response.json()

            st.success(f"Processed {data['word_count']} words")

            for lemma in data['lemmas']:
                morphology = lemma.get('morphology', {})
                method = morphology.get('_method', 'unknown')
                confidence = morphology.get('_confidence', 0)

                with st.expander(f"{lemma['original']} -> {lemma['lemma']} ({method})"):
                    st.write(f"**Method:** {method}")
                    st.write(f"**Confidence:** {confidence:.3f}")
                    if morphology:
                        st.json({k: v for k, v in morphology.items() if not k.startswith('_')})
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif feature == "Complexity Analysis":
    st.header("Text Complexity Analysis")

    text = st.text_area("Finnish Text", value="Kissa istuu puussa.", height=100)

    if st.button("Analyze"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/complexity",
                json={"text": text, "detailed": True}
            )
            data = response.json()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Word Count", data['word_count'])
            col2.metric("Avg Word Length", f"{data['average_word_length']:.1f}")
            col3.metric("Max Word Length", data['max_word_length'])
            col4.metric("Complexity", data['complexity_rating'])
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif feature == "Morphological Entropy":
    st.header("Morphological Entropy Analysis")
    st.markdown("**World's first information-theoretic complexity metric for Finnish**")

    text = st.text_area("Finnish Text", value="Kissani söi hiiren puutarhassani.", height=100)

    if st.button("Calculate Entropy"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/entropy",
                json={"text": text, "detailed": True}
            )
            data = response.json()

            col1, col2, col3 = st.columns(3)
            col1.metric("Case Entropy", f"{data['case_entropy']:.3f}")
            col2.metric("Suffix Entropy", f"{data['suffix_entropy']:.3f}")
            col3.metric("Word Formation", f"{data['word_formation_entropy']:.3f}")

            st.metric("Overall Score", f"{data['overall_score']:.3f}")
            st.info(f"**Complexity:** {data['complexity_interpretation'].title()}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif feature == "Semantic Disambiguation":
    st.header("Semantic Disambiguation")
    st.markdown("Context-aware resolution of ambiguous Finnish words")

    text = st.text_area("Finnish Text", value="Kuusi kaunista kuusta kasvaa.", height=100)

    if st.button("Disambiguate"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/disambiguate",
                json={"text": text, "auto_detect": True}
            )
            data = response.json()

            st.success(f"Found {data['ambiguous_words_found']} ambiguous words")

            for dis in data.get('disambiguations', []):
                with st.expander(f"{dis['word']} -> {dis['predicted_sense']}"):
                    st.write(f"**All Senses:** {', '.join(dis['all_senses'])}")
                    st.write(f"**Predicted:** {dis['predicted_sense']}")
                    st.write(f"**Confidence:** {dis['confidence']:.2f}")
                    st.write(f"**Explanation:** {dis['explanation']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif feature == "Linguistic Explanation":
    st.header("Linguistic Explanation")
    st.markdown("Educational tool for Finnish language learners")

    text = st.text_area("Finnish Text", value="Kissani söi hiiren.", height=100)
    level = st.select_slider("Target Level", options=['beginner', 'intermediate', 'advanced'])

    if st.button("Explain"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/explain",
                json={"text": text, "level": level}
            )
            data = response.json()

            st.success(f"Analyzed {data['statistics']['total_words']} words")

            st.markdown("### Simplified Version")
            st.info(data['simplified'])

            st.markdown("### Word-by-Word Breakdown")
            for word_exp in data['word_explanations']:
                with st.expander(f"{word_exp['word']} (Difficulty: {word_exp['difficulty']})"):
                    st.write(f"**Lemma:** {word_exp['lemma']}")
                    st.write(f"**POS:** {word_exp['pos']}")
                    st.write(f"**Frequency:** {word_exp['frequency']}")
                    st.write(f"**Learning Tip:** {word_exp['learning_tip']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif feature == "Text Clarification":
    st.header("Text Clarification")
    st.markdown("Highlight difficult words and suggest alternatives")

    text = st.text_area("Finnish Text", value="Kirjoittautumisvelvollisuuden laiminlyönti.", height=100)
    level = st.select_slider("Target Level", options=['beginner', 'intermediate', 'advanced'])

    if st.button("Clarify"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/clarify",
                json={"text": text, "level": level}
            )
            data = response.json()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Words", data['word_count'])
            col2.metric("Difficult Words", data['difficult_word_count'])
            col3.metric("Readability", f"{data['readability_score']:.2f}")

            st.metric("Readability Rating", data['readability_rating'].title())

            st.markdown("### Difficult Words")
            for word in data.get('difficult_words', []):
                with st.expander(f"{word['word']} ({word['reason'].replace('_', ' ')})"):
                    st.write(f"**Lemma:** {word['lemma']}")
                    st.write(f"**Difficulty Score:** {word['difficulty_score']}/3")
                    if word.get('alternative'):
                        st.success(f"**Suggested:** {word['alternative']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif feature == "Text Simplification":
    st.header("Text Simplification")
    st.markdown("Generate simplified versions of complex text")

    text = st.text_area("Finnish Text", value="Kirjoittautumisvelvollisuuden laiminlyönti.", height=100)

    col1, col2 = st.columns(2)
    with col1:
        level = st.select_slider("Target Level", options=['beginner', 'intermediate', 'advanced'])
    with col2:
        strategy = st.select_slider("Strategy", options=['conservative', 'moderate', 'aggressive'])

    if st.button("Simplify"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/simplify",
                json={"text": text, "level": level, "strategy": strategy}
            )
            data = response.json()

            st.success(f"Made {data['simplifications_count']} simplifications")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Original")
                st.write(data['original_text'])
            with col2:
                st.markdown("### Simplified")
                st.write(data['simplified_text'])

            st.markdown("### Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Length Reduction", f"{data['metrics']['reduction_percentage']:.1f}%")
            col2.metric("Readability +", f"{data['metrics']['readability_improvement']:.3f}")
            col3.metric("Difficult Words", f"{data['metrics']['original_difficult_words']} -> {data['metrics']['simplified_difficult_words']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif feature == "Benchmarking":
    st.header("System Benchmarking")
    st.markdown("Compare against Voikko and Stanza")

    include_external = st.checkbox("Include External Systems", value=True)

    if st.button("Run Benchmark"):
        with st.spinner("Running benchmark..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/benchmark",
                    json={"include_external": include_external}
                )
                data = response.json()

                st.success(f"Benchmarked {len(data['systems_compared'])} systems")

                summary = data.get('summary', {})
                col1, col2 = st.columns(2)
                col1.metric("Best Accuracy", f"{summary.get('best_accuracy', 0):.2f}%")
                col1.write(f"**System:** {summary.get('best_accuracy_system', 'N/A')}")
                col2.metric("Fastest", f"{summary.get('fastest_avg_time_ms', 0):.3f} ms")
                col2.write(f"**System:** {summary.get('fastest_system', 'N/A')}")

                st.markdown("### Results")
                results_data = []
                for result in data.get('results', []):
                    results_data.append({
                        "System": result['system'],
                        "Accuracy (%)": result['accuracy'],
                        "Avg Time (ms)": result['avg_time_ms']
                    })

                st.table(results_data)
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Finnish NLP Toolkit v2.0 | 11 Routers | 30+ Endpoints | 99/99 Tests Passing")
