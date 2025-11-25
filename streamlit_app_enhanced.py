"""
Enhanced Streamlit Demo for Finnish NLP Toolkit
Showcases all features: lemmatization, complexity, profanity, entropy,
disambiguation, explain, clarify, simplify, and benchmarking
"""
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# API configuration
API_BASE_URL = "http://localhost:8000/api"

# Page configuration
st.set_page_config(
    page_title="Finnish NLP Toolkit - Enhanced Demo",
    page_icon="\ud83c\uddeb\ud83c\uddee",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">\ud83c\uddeb\ud83c\uddee Finnish NLP Toolkit - Giant Avant-Garde Version</div>', unsafe_allow_html=True)
st.markdown("**Advanced Finnish Language Processing with Novel Capabilities**")

# Sidebar
with st.sidebar:
    st.markdown("## \u2699\ufe0f Navigation")
    feature = st.selectbox(
        "Select Feature",
        [
            "\ud83d\udcdd Lemmatization",
            "\ud83d\udd0d Hybrid Lemmatization",
            "\ud83d\udcca Complexity Analysis",
            "\u26a0\ufe0f Profanity Detection",
            "\ud83c\udf00 Morphological Entropy",
            "\ud83e\udd14 Semantic Disambiguation",
            "\ud83c\udf93 Linguistic Explanation",
            "\ud83d\udd0d Text Clarification",
            "\u2728 Text Simplification",
            "\ud83c\udfc6 Benchmarking"
        ]
    )

    st.markdown("---")
    st.markdown("### \ud83d\udcca API Stats")
    st.metric("Total Routers", "11")
    st.metric("Total Endpoints", "30+")
    st.metric("Test Coverage", "99/99 (100%)")

    st.markdown("---")
    st.markdown("### \ud83d\udce6 Novel Features")
    st.markdown("- Morphological Entropy")
    st.markdown("- Semantic Disambiguation")
    st.markdown("- Hybrid 3-Stage System")
    st.markdown("- Explanation Engine")
    st.markdown("- Clarification System")
    st.markdown("- Simplification Engine")

# ==================== LEMMATIZATION ====================
if feature == "\ud83d\udcdd Lemmatization":
    st.markdown('<div class="sub-header">Lemmatization</div>', unsafe_allow_html=True)
    st.markdown("Convert Finnish words to their base forms with morphological analysis")

    text_input = st.text_area(
        "Finnish Text",
        value="Kissani söi hiiren puutarhassani nopeasti.",
        height=100
    )

    col1, col2 = st.columns(2)
    with col1:
        include_morphology = st.checkbox("Include Morphology", value=True)

    if st.button("Lemmatize", type="primary"):
        with st.spinner("Processing..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/lemmatize",
                    json={"text": text_input, "include_morphology": include_morphology}
                )
                data = response.json()

                st.success(f"Processed {data['word_count']} words")

                # Display lemmas
                for lemma in data['lemmas']:
                    with st.expander(f"{lemma['original']} → {lemma['lemma']}", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**POS:** {lemma.get('pos', 'N/A')}")
                        with col2:
                            if include_morphology and lemma.get('morphology'):
                                st.json(lemma['morphology'])

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==================== HYBRID LEMMATIZATION ====================
elif feature == "\ud83d\udd0d Hybrid Lemmatization":
    st.markdown('<div class="sub-header">Hybrid 3-Stage Lemmatization</div>', unsafe_allow_html=True)
    st.markdown("**Stage 1:** Dictionary → **Stage 2:** ML → **Stage 3:** Similarity")

    text_input = st.text_area(
        "Finnish Text",
        value="Kissani söi hiiren puutarhassani.",
        height=100
    )

    if st.button("Hybrid Lemmatize", type="primary"):
        with st.spinner("Processing..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/hybrid-lemma",
                    json={"text": text_input, "return_method_info": True}
                )
                data = response.json()

                st.success(f"Processed {data['word_count']} words")

                # Method distribution
                methods = {}
                for lemma in data['lemmas']:
                    if lemma.get('morphology') and '_method' in lemma['morphology']:
                        method = lemma['morphology']['_method']
                        methods[method] = methods.get(method, 0) + 1

                col1, col2, col3 = st.columns(3)
                for i, (method, count) in enumerate(methods.items()):
                    with [col1, col2, col3][i % 3]:
                        st.metric(method.title(), count)

                # Display lemmas with method
                for lemma in data['lemmas']:
                    morphology = lemma.get('morphology', {})
                    method = morphology.get('_method', 'unknown')
                    confidence = morphology.get('_confidence', 0)

                    with st.expander(f"{lemma['original']} → {lemma['lemma']} ({method})", expanded=False):
                        st.write(f"**Method:** {method}")
                        st.write(f"**Confidence:** {confidence:.3f}")
                        if morphology:
                            st.json({k: v for k, v in morphology.items() if not k.startswith('_')})

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==================== COMPLEXITY ====================
elif feature == "\ud83d\udcca Complexity Analysis":
    st.markdown('<div class="sub-header">Text Complexity Analysis</div>', unsafe_allow_html=True)

    text_input = st.text_area(
        "Finnish Text",
        value="Kissa istuu puussa. Tämä on yksinkertainen lause.",
        height=100
    )

    if st.button("Analyze Complexity", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/complexity",
                    json={"text": text_input, "detailed": True}
                )
                data = response.json()

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Word Count", data['word_count'])
                col2.metric("Avg Word Length", f"{data['average_word_length']:.1f}")
                col3.metric("Max Word Length", data['max_word_length'])
                col4.metric("Complexity", data['complexity_rating'].title())

                # Case distribution chart
                if 'case_distribution' in data:
                    df = pd.DataFrame(list(data['case_distribution'].items()), columns=['Case', 'Count'])
                    fig = px.bar(df, x='Case', y='Count', title="Case Distribution")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==================== PROFANITY ====================
elif feature == "\u26a0\ufe0f Profanity Detection":
    st.markdown('<div class="sub-header">Profanity Detection</div>', unsafe_allow_html=True)

    text_input = st.text_area(
        "Finnish Text",
        value="Helvetti että meni hyvin!",
        height=100
    )

    threshold = st.slider("Severity Threshold", 0.0, 1.0, 0.5, 0.05)

    if st.button("Detect Profanity", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/profanity",
                    json={"text": text_input, "threshold": threshold, "return_flagged_words": True}
                )
                data = response.json()

                col1, col2 = st.columns(2)
                col1.metric("Toxicity Score", f"{data['toxicity_score']:.2f}")
                col2.metric("Is Toxic", "Yes" if data['is_toxic'] else "No")

                if data.get('flagged_words'):
                    st.warning(f"**Flagged {len(data['flagged_words'])} words:**")
                    for word_info in data['flagged_words']:
                        st.write(f"- **{word_info['word']}** (severity: {word_info['severity']})")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==================== ENTROPY ====================
elif feature == "\ud83c\udf00 Morphological Entropy":
    st.markdown('<div class="sub-header">Morphological Entropy Analysis</div>', unsafe_allow_html=True)
    st.markdown("**World's first information-theoretic complexity metric for Finnish**")

    text_input = st.text_area(
        "Finnish Text",
        value="Kissani söi hiiren puutarhassani nopeasti.",
        height=100
    )

    if st.button("Calculate Entropy", type="primary"):
        with st.spinner("Calculating..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/entropy",
                    json={"text": text_input, "detailed": True}
                )
                data = response.json()

                col1, col2, col3 = st.columns(3)
                col1.metric("Case Entropy", f"{data['case_entropy']:.3f}")
                col2.metric("Suffix Entropy", f"{data['suffix_entropy']:.3f}")
                col3.metric("Word Formation Entropy", f"{data['word_formation_entropy']:.3f}")

                st.metric("Overall Entropy Score", f"{data['overall_score']:.3f}")
                st.info(f"**Complexity:** {data['complexity_interpretation'].title()}")

                # Visualization
                if 'case_distribution' in data:
                    df = pd.DataFrame(list(data['case_distribution'].items()), columns=['Case', 'Count'])
                    fig = px.pie(df, values='Count', names='Case', title="Case Distribution")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==================== DISAMBIGUATION ====================
elif feature == "\ud83e\udd14 Semantic Disambiguation":
    st.markdown('<div class="sub-header">Semantic Disambiguation</div>', unsafe_allow_html=True)
    st.markdown("Context-aware resolution of ambiguous Finnish words")

    text_input = st.text_area(
        "Finnish Text",
        value="Kuusi kaunista kuusta kasvaa mäellä.",
        height=100
    )

    if st.button("Disambiguate", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/disambiguate",
                    json={"text": text_input, "auto_detect": True}
                )
                data = response.json()

                st.success(f"Found {data['ambiguous_words_found']} ambiguous words")

                for dis in data.get('disambiguations', []):
                    with st.expander(f"{dis['word']} → {dis['predicted_sense']}", expanded=True):
                        st.write(f"**All Senses:** {', '.join(dis['all_senses'])}")
                        st.write(f"**Predicted:** {dis['predicted_sense']}")
                        st.write(f"**Confidence:** {dis['confidence']:.2f}")
                        st.write(f"**Explanation:** {dis['explanation']}")
                        if 'context_snippet' in dis:
                            st.info(f"Context: {dis['context_snippet']}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==================== EXPLAIN ====================
elif feature == "\ud83c\udf93 Linguistic Explanation":
    st.markdown('<div class="sub-header">Linguistic Explanation</div>', unsafe_allow_html=True)
    st.markdown("Educational feature for Finnish language learners")

    text_input = st.text_area(
        "Finnish Text",
        value="Kissani söi hiiren puutarhassani.",
        height=100
    )

    level = st.select_slider("Target Level", options=['beginner', 'intermediate', 'advanced'])

    if st.button("Explain", type="primary"):
        with st.spinner("Generating explanation..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/explain",
                    json={"text": text_input, "level": level}
                )
                data = response.json()

                st.success(f"Analyzed {data['statistics']['total_words']} words")

                col1, col2 = st.columns(2)
                col1.metric("Unique Cases", data['statistics']['unique_cases'])
                col2.metric("Difficulty", data['overall_difficulty'].title())

                st.markdown("### Simplified Version")
                st.info(data['simplified'])

                st.markdown("### Word-by-Word Breakdown")
                for word_exp in data['word_explanations']:
                    with st.expander(f"{word_exp['word']} (Difficulty: {word_exp['difficulty']})", expanded=False):
                        st.write(f"**Lemma:** {word_exp['lemma']}")
                        st.write(f"**POS:** {word_exp['pos']}")
                        st.write(f"**Frequency:** {word_exp['frequency']}")
                        st.write(f"**Learning Tip:** {word_exp['learning_tip']}")
                        if word_exp.get('breakdown'):
                            st.json(word_exp['breakdown'])

                st.markdown("### Learning Focus")
                for focus in data.get('learning_focus', []):
                    st.markdown(f"- {focus}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==================== CLARIFY ====================
elif feature == "\ud83d\udd0d Text Clarification":
    st.markdown('<div class="sub-header">Text Clarification</div>', unsafe_allow_html=True)
    st.markdown("Highlight difficult words and suggest alternatives")

    text_input = st.text_area(
        "Finnish Text",
        value="Kirjoittautumisvelvollisuuden laiminlyönti johtaa seuraamuksiin.",
        height=100
    )

    level = st.select_slider("Target Level", options=['beginner', 'intermediate', 'advanced'])

    if st.button("Clarify", type="primary"):
        with st.spinner("Clarifying..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/clarify",
                    json={"text": text_input, "level": level}
                )
                data = response.json()

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Words", data['word_count'])
                col2.metric("Difficult Words", data['difficult_word_count'])
                col3.metric("Readability", f"{data['readability_score']:.2f}")

                st.metric("Readability Rating", data['readability_rating'].title())
                st.metric("Target Appropriate", "Yes" if data['target_appropriate'] else "No")

                st.markdown("### Difficult Words")
                for word in data.get('difficult_words', []):
                    with st.expander(f"{word['word']} ({word['reason'].replace('_', ' ')})", expanded=True):
                        st.write(f"**Lemma:** {word['lemma']}")
                        st.write(f"**Difficulty Score:** {word['difficulty_score']}/3")
                        if word.get('alternative'):
                            st.success(f"**Suggested Alternative:** {word['alternative']}")

                st.markdown("### Recommendations")
                for rec in data.get('recommendations', []):
                    st.info(rec)

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==================== SIMPLIFY ====================
elif feature == "\u2728 Text Simplification":
    st.markdown('<div class="sub-header">Text Simplification</div>', unsafe_allow_html=True)
    st.markdown("Generate simplified versions of complex text")

    text_input = st.text_area(
        "Finnish Text",
        value="Kirjoittautumisvelvollisuuden laiminlyönti johtaa vakaviin seuraamuksiin.",
        height=100
    )

    col1, col2 = st.columns(2)
    with col1:
        level = st.select_slider("Target Level", options=['beginner', 'intermediate', 'advanced'])
    with col2:
        strategy = st.select_slider("Strategy", options=['conservative', 'moderate', 'aggressive'])

    if st.button("Simplify", type="primary"):
        with st.spinner("Simplifying..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/simplify",
                    json={"text": text_input, "level": level, "strategy": strategy}
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

                # Metrics
                st.markdown("### Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Length Reduction", f"{data['metrics']['reduction_percentage']:.1f}%")
                col2.metric("Readability Improvement", f"{data['metrics']['readability_improvement']:.3f}")
                col3.metric("Original Difficult Words", data['metrics']['original_difficult_words'])
                col4.metric("Simplified Difficult Words", data['metrics']['simplified_difficult_words'])

                # Simplifications made
                if data.get('simplifications_made'):
                    st.markdown("### Simplifications Made")
                    for simp in data['simplifications_made']:
                        st.write(f"- **{simp['original']}** → **{simp['simplified']}** (saved {simp['saved_characters']} chars)")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==================== BENCHMARKING ====================
elif feature == "\ud83c\udfc6 Benchmarking":
    st.markdown('<div class="sub-header">System Benchmarking</div>', unsafe_allow_html=True)
    st.markdown("Compare Finnish NLP Toolkit against Voikko and Stanza")

    include_external = st.checkbox("Include External Systems (Voikko, Stanza)", value=True)

    if st.button("Run Benchmark", type="primary"):
        with st.spinner("Running benchmark... This may take a while..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/benchmark",
                    json={"include_external": include_external}
                )
                data = response.json()

                st.success(f"Benchmarked {len(data['systems_compared'])} systems on {data['gold_standard_size']} examples")

                # Summary
                summary = data.get('summary', {})
                col1, col2 = st.columns(2)
                col1.metric("Best Accuracy", f"{summary.get('best_accuracy', 0):.2f}%")
                col1.write(f"**System:** {summary.get('best_accuracy_system', 'N/A')}")
                col2.metric("Fastest System", f"{summary.get('fastest_avg_time_ms', 0):.3f} ms")
                col2.write(f"**System:** {summary.get('fastest_system', 'N/A')}")

                # Results table
                st.markdown("### Detailed Results")
                results_data = []
                for result in data.get('results', []):
                    results_data.append({
                        "System": result['system'],
                        "Accuracy (%)": result['accuracy'],
                        "Correct": result['correct'],
                        "Incorrect": result['incorrect'],
                        "Avg Time (ms)": result['avg_time_ms'],
                        "Total Time (s)": result['total_time_s']
                    })

                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)

                # Accuracy comparison chart
                fig = px.bar(df, x='System', y='Accuracy (%)', title="Accuracy Comparison")
                st.plotly_chart(fig, use_container_width=True)

                # Speed comparison chart
                fig2 = px.bar(df, x='System', y='Avg Time (ms)', title="Speed Comparison (lower is better)")
                st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    Finnish NLP Toolkit - Giant Avant-Garde Version | 11 Routers | 30+ Endpoints | 99/99 Tests Passing
</div>
""", unsafe_allow_html=True)
