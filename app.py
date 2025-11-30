import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Using default English fonts, no special Matplotlib config is needed for Streamlit Cloud.
# We keep axes.unicode_minus=False just in case, but it's often not needed for English plots.
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. Core Mathematical Model Functions ---

def generate_confession_times(mode, n=50):
    i_series = np.array(range(1, n + 1))
    if mode == "mo_ceng":      
        return np.array([1 + 1/i for i in i_series])
    elif mode == "sao_dong":   
        return np.array([1 - 1/i for i in i_series])
    else:
        return np.sort(np.random.uniform(0, 10, n))

def is_brave(times):
    if len(times) < 5: return False
    diff = np.abs(np.diff(times[-5:]))
    return np.all(diff < 1e-3)

def success_rate(t, A, t0, sigma):
    if sigma <= 0: sigma = 1e-5
    return A * np.exp(-((t - t0)**2) / (2*sigma**2))

def stability_analysis(t, A_val, t0, sigma, delta=0.01):
    right_limit = success_rate(t + delta, A_val, t0, sigma)
    left_limit = success_rate(t - delta, A_val, t0, sigma)

    if np.isnan(left_limit) or np.isnan(right_limit):
        return "Self-sabotage 💀"

    is_limit_equal = abs(left_limit - right_limit) < 1e-2

    if is_limit_equal:
        if abs(left_limit - success_rate(t, A_val, t0, sigma)) < 1e-2:
            return "Developing 🌱"
        else:
            return "Fickle 🍃"
    else:
        return "On Track 🎁"

def determine_mode(delay_choice, change_choice):
    if delay_choice == 1 and change_choice == 1:
        return "mo_ceng (Delaying)"
    elif delay_choice == 2 or change_choice == 2:
        return "sao_dong (Impulsive)"
    else:
        return "random (Random)"

# --- 2. Helper Functions: Scoring and Classification ---

def calculate_score(raw_scores):
    total_score = sum(raw_scores)
    # Map raw score (3-15) to final score (1-10)
    final_score = 1 + ((total_score - 3) / (15 - 3)) * (10 - 1)
    return np.clip(round(final_score), 1, 10) 

def classify_love_type(I, P, C, threshold=7):
    is_i = I >= threshold
    is_p = P >= threshold
    is_c = C >= threshold

    if is_i and is_p and is_c:
        return "Consummate Love", "Consummate Love: The ideal state, all three components are present."
    elif is_i and is_c:
        return "Companionate Love", "Companionate Love: Stable and deep affection, but passion may have faded."
    elif is_p and is_c:
        return "Fatuous Love", "Fatuous Love: Whirlwind relationship lacking deep intimacy."
    elif is_i and is_p:
        return "Romantic Love", "Romantic Love: Deep intimacy and passion, but lacks long-term commitment."
    elif is_i:
        return "Liking (Friendship)", "Liking: Contains only intimacy, true friendship."
    elif is_p:
        return "Infatuation", "Infatuation: Contains only passion, often love at first sight or one-sided."
    elif is_c:
        return "Empty Love", "Empty Love: Contains only commitment, lacking intimacy and attraction."
    else:
        return "Non-love", "Non-love: None of the three components are met, starting from zero."

# --- 3. Visualization Functions ---

@st.cache_data
def plot_love_triangle(I, P, C):
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    
    labels = ['Intimacy (I)', 'Passion (P)', 'Commitment (C)']
    values = np.array([I, P, C])
    values = np.concatenate((values, [I]))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    plot_color = 'mediumvioletred' 
    fill_color = 'lightpink'
    
    ax.plot(angles, values, 'o-', linewidth=3, color=plot_color, markerfacecolor=plot_color, markersize=8, label="Current Relationship Status")
    ax.fill(angles, values, color=fill_color, alpha=0.6)

    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels, fontsize=12, color='darkslategray')
    ax.set_ylim(0, 10) 
    ax.set_yticks(np.arange(0, 11, 2)) 
    ax.tick_params(axis='y', colors='gray', labelsize=10)
    ax.spines['polar'].set_visible(False) 
    ax.grid(color='lightgray', linestyle='--')

    love_type, description = classify_love_type(I, P, C)
    ax.text(0, 0, f"Type: {love_type}\n\n{description}", 
            ha='center', va='center', fontsize=11, color=plot_color, 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle="round,pad=0.7"))

    ax.set_title("💞 Sternberg's Triangle of Love: Relationship Type Analysis", va='bottom', fontsize=16, pad=15, color='darkslategray')
    return fig

@st.cache_data
def plot_success_curve(A, t_peak, sigma, current_time):
    t_start = max(0, min(t_peak, current_time) - 2 * sigma)
    t_end = max(10, max(t_peak, current_time) + 2 * sigma)
    t = np.linspace(t_start, t_end, 300) 
    p = success_rate(t, A, t_peak, sigma)
    p = np.clip(p, 0, 1)
    predicted_rate = success_rate(current_time, A, t_peak, sigma)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    ax.fill_between(t, 0, p, color='skyblue', alpha=0.2, label="Success Rate Area")
    ax.plot(t, p, color='steelblue', linewidth=3, label="Proposal Success Rate p(t)")
    
    ax.axvline(current_time, color='darkorange', linestyle='-', linewidth=2, label=f"Predicted Action T={current_time:.2f} Weeks")
    ax.scatter(current_time, predicted_rate, s=150, color='darkorange', zorder=5, marker='o', edgecolor='white', linewidth=2)
    
    ax.axvline(t_peak, color='crimson', linestyle='--', linewidth=1.5, label=f"Actual Peak Tpeak={t_peak:.2f} Weeks")
    ax.axhline(A, color='forestgreen', linestyle=':', label=f"Max Success Rate A={A:.2f}", linewidth=1.5)

    ax.annotate(f"Predicted Rate: {predicted_rate:.2f}", 
                 xy=(current_time, predicted_rate), 
                 xytext=(current_time + 0.5 * sigma, predicted_rate - 0.1),
                 arrowprops=dict(facecolor='darkorange', shrink=0.05, width=1, headwidth=8, headlength=8, alpha=0.7),
                 fontsize=11, color='darkorange')

    ax.set_xlabel("Time t (Weeks)", fontsize=13)
    ax.set_ylabel("Success Rate p(t)", fontsize=13)
    ax.set_title("📈 Romance Timing Analysis: Proposal Success Curve", fontsize=16, pad=15)
    ax.legend(fontsize=10)
    
    return fig

# --- 4. Streamlit Main Program ---

def run_analysis(data):
    # Extract data
    q1_delay = data['q1_delay']
    q2_change = data['q2_change']
    raw_i = [data[f'i{i}'] for i in range(1, 4)]
    raw_p = [data[f'p{i}'] for i in range(1, 4)]
    raw_c = [data[f'c{i}'] for i in range(1, 4)]
    t0_ideal = data['t0_weeks']
    
    # 1. Behavioral Mode
    mode = determine_mode(q1_delay, q2_change)
    
    # 2. IPC Scoring
    I = calculate_score(raw_i)
    P = calculate_score(raw_p)
    C = calculate_score(raw_c)

    # 3. Calculate A, sigma, t_peak
    A = 0.5 + ((I + P + C) / 30.0) * 0.5 
    sigma = 0.5 + (C / 10.0) * 1.5       
    
    I_norm = I / 10.0
    C_norm = C / 10.0
    alpha = 1.0 - ((I_norm + C_norm) / 2.0) * 0.5
    
    t_peak = t0_ideal * alpha
    t_peak = np.clip(t_peak, 0.01, None) 

    # 4. Calculate Predicted Time t
    times = generate_confession_times(mode.split(' ')[0]) # Use English mode part
    brave = is_brave(times)
    mean_times_last = np.mean(times[-10:])
    
    if mode.startswith("random"):
        current_time_mapped = t_peak + (mean_times_last - np.mean(times)) * (sigma / 4)
    else:
        current_time_mapped = t_peak + (mean_times_last - 1) * (sigma / 2)
    
    current_time_mapped = np.clip(current_time_mapped, 0.01, t_peak + sigma * 3)
    
    # 5. Status Analysis
    status = stability_analysis(current_time_mapped, A, t_peak, sigma)
    predicted_rate = success_rate(current_time_mapped, A, t_peak, sigma)
    love_type, _ = classify_love_type(I, P, C)

    # --- Results Display ---
    st.markdown("## ✅ **Relationship Analysis Report**")
    st.markdown(f"### Current Relationship Status: **{status}**")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Relationship Core Analysis (IPC)")
        st.metric(label="**Intimacy (I) Score**", value=f"{I}/10")
        st.metric(label="**Passion (P) Score**", value=f"{P}/10")
        st.metric(label="**Commitment (C) Score**", value=f"{C}/10")
        st.markdown(f"**Love Type:** *{love_type}*")
        st.markdown(f"**Max Success Rate (A):** {A:.2f}")

    with col2:
        st.subheader("🧭 Timing Analysis (T)")
        st.metric(label="**Ideal Anchor Moment T₀**", value=f"{t0_ideal:.2f} Weeks")
        st.metric(label="**🌟 Actual Peak Moment Tpeak**", value=f"{t_peak:.2f} Weeks")
        st.metric(label="**Predicted Action Moment T**", value=f"{current_time_mapped:.2f} Weeks", delta=f"Deviation from Peak {current_time_mapped - t_peak:.2f} Weeks")
        st.metric(label="**Predicted Success Rate p(T)**", value=f"{predicted_rate:.2f}")
        st.markdown(f"**Behavioral Mode:** {mode}")
        st.markdown(f"**Is Brave to Propose:** {'✅ Yes' if brave else '❌ No'}")

    st.markdown("---")
    st.subheader("💞 Triangle of Love Plot")
    st.pyplot(plot_love_triangle(I, P, C))

    st.subheader("📈 Proposal Success Curve")
    st.pyplot(plot_success_curve(A, t_peak, sigma, current_time_mapped))
    
    st.markdown("---")


def main():
    st.title("💌 Love Emergency · Proposal Analysis System")
    st.markdown("Please complete the following questionnaire. The system will calculate your best proposal timing based on your relationship and behavioral patterns.")

    # Use Streamlit session state to save questionnaire data
    if 'analysis_data' not in st.session_state:
        st.session_state['analysis_data'] = None

    with st.form("love_analysis_form"):
        # --- 1. Behavioral Tendency Questionnaire ---
        st.subheader("1. 📝 Behavioral Tendency Questionnaire")
        q1_delay = st.radio(
            "Q1. Assuming you propose, would you prefer to:",
            options=[1, 2],
            format_func=lambda x: "Delay (1)" if x == 1 else "Advance (2)",
            index=0,
            key='q1_delay'
        )
        q2_change = st.radio(
            "Q2. Your proposal plan is to:",
            options=[1, 2],
            format_func=lambda x: "Not easily change it (1)" if x == 1 else "Repeatedly revise it (2)",
            index=0,
            key='q2_change'
        )

        # --- 2. Relationship Assessment (IPC) ---
        st.subheader("2. 💖 Relationship Assessment (1-5 points, 5 is Strongly Agree)")
        
        ipc_scores = {}
        
        st.markdown("##### [Intimacy (I)]")
        ipc_scores['i1'] = st.slider("Q3. I can share my deepest fears and secrets with my partner.", 1, 5, 3, key='i1')
        ipc_scores['i2'] = st.slider("Q4. When facing difficulties, my partner is my first choice for support.", 1, 5, 3, key='i2')
        ipc_scores['i3'] = st.slider("Q5. When we are together, we often feel a sense of 'telepathy' or deep understanding.", 1, 5, 3, key='i3')
        
        st.markdown("##### [Passion (P)]")
        ipc_scores['p1'] = st.slider("Q6. When thinking about or seeing my partner, I feel excitement and a racing heart.", 1, 5, 3, key='p1')
        ipc_scores['p2'] = st.slider("Q7. I actively try to create romance and surprises to keep things fresh.", 1, 5, 3, key='p2')
        ipc_scores['p3'] = st.slider("Q8. I initiate or desire physical contact or intimate behavior with my partner.", 1, 5, 3, key='p3')

        st.markdown("##### [Commitment (C)]")
        ipc_scores['c1'] = st.slider("Q9. I have a clear long-term plan for this relationship (e.g., more than a year).", 1, 5, 3, key='c1')
        ipc_scores['c2'] = st.slider("Q10. Even if we disagree, I will stick with this relationship rather than easily giving up.", 1, 5, 3, key='c2')
        ipc_scores['c3'] = st.slider("Q11. I believe my partner is the 'only' choice worth my time and effort.", 1, 5, 3, key='c3')

        # --- 3. Key Moment T₀ Guidance ---
        st.subheader("3. 🧭 Key Moment T₀ Guidance")
        t0_type = st.selectbox(
            "Select the type of 'Key Event' you consider ideal:",
            options=["Anniversary/Milestone", "Personal Event/Holiday", "Emotional Peak"],
            key='t0_type'
        )
        t0_weeks = st.number_input(
            f"Enter how many **weeks** away is this '{t0_type}' event? (e.g.: 3.5)",
            min_value=0.1,
            value=4.0,
            step=0.1,
            key='t0_weeks'
        )
        
        submitted = st.form_submit_button("🚀 Get My Relationship Analysis Report")

    if submitted:
        # Collect all data and save to session_state
        analysis_data = {
            'q1_delay': q1_delay,
            'q2_change': q2_change,
            'i1': ipc_scores['i1'], 'i2': ipc_scores['i2'], 'i3': ipc_scores['i3'],
            'p1': ipc_scores['p1'], 'p2': ipc_scores['p2'], 'p3': ipc_scores['p3'],
            'c1': ipc_scores['c1'], 'c2': ipc_scores['c2'], 'c3': ipc_scores['c3'],
            't0_weeks': t0_weeks,
            't0_type': t0_type
        }
        st.session_state['analysis_data'] = analysis_data
        
    if st.session_state['analysis_data']:
        run_analysis(st.session_state['analysis_data'])

if __name__ == '__main__':
    main()
