# Knowledge Integrity & Expert Verification Engine 

An automated Reinforcement Learning (RL) gatekeeper designed to detect and flag "Expertise Fraud" in technical hiring pipelines. 

This proof-of-concept system evaluates the probability that a candidate is utilizing Large Language Models (LLMs) like ChatGPT to feign technical depth by analyzing historical profile anomalies, verifiable web signals, and the statistical linguistic footprints of their screening answers.

---

## The Problem
In the era of ubiquitous AI, technical hiring faces a new threat. Fraudulent actors leverage tools like ChatGPT to construct highly convincing resumes and perfectly draft screening responses for high-demand roles they never actually held. Passing these fabricated experts through the hiring funnel wastes incredibly expensive senior engineering time during live technical interviews.

## The Solution
This project introduces a custom Reinforcement Learning (RL) agent that acts as an automated filter. Instead of a standard binary classifier, the system uses a custom policy network optimized via an asymmetric reward matrix. The agent aggressively penalizes False Negatives (wasting organization's time and resources on a fraud) while rewarding True Positives.

### Feature Engineering
To differentiate AI-assisted fabrication from genuine career growth, the environment extracts a 5-dimensional continuous state space for each candidate:

1. **`Perplexity_Score`**: Measures word predictability in screening answers. (LLMs exhibit abnormally low perplexity).
2. **`Burstiness_Score`**: Measures variance in sentence length and structure.
3. **`Skill_Velocity_Index`**: Evaluates the rate of acquiring complex skills over time (flagging "Zero to Hero" overnight jumps).
4. **`Semantic_Delta`**: Measures the NLP semantic distance (jargon complexity jumps) between historical profile snapshots.
5. **`Digital_Footprint_Density`**: Correlates claimed years of experience with verifiable, time-stamped external activity (e.g., GitHub commits, Stack Overflow age).

---

## RL Architecture & Training

### Custom PyTorch Policy Network
The agent is built using PyTorch. Because this environment functions as a continuous Contextual Bandit rather than a sequential Markov Decision Process (MDP), standard RL loss algorithms (like basic REINFORCE) lead to gradient collapse. 

To solve this, the model uses a **Custom Differentiable Negative-Reward Loss**. The agent outputs a probability $p \in [0.0, 1.0]$. The loss function directly minimizes the negative expected reward based on our asymmetric business logic:

* **When Candidate is a Fraud (Ground Truth = 1):** * Maximum Reward (+10) for $p=1.0$
  * Maximum Penalty (-20) for $p=0.0$ (Severe penalty to protect engineering time)
  * *Loss Function:* `-(30 * p - 20)`
* **When Candidate is a Real Expert (Ground Truth = 0):**
  * Maximum Reward (+2) for $p=0.0$
  * Moderate Penalty (-5) for $p=1.0$
  * *Loss Function:* `-(7 * (1 - p) - 5)`

### Results & Evaluation
The model was trained on a synthetic dataset of 1,000 candidate profiles with realistic noise distributions. Evaluated using a strict 90/10 Train/Test split (with `torch.no_grad()` enabled):
* **Test Set Performance:** The agent successfully generalized to unseen data, capturing **~86% of the theoretical maximum reward**.

---

## Future Scope: Multi-Modal "Live" Evaluator
The next evolution of this model is to deploy it as a real-time "Co-Pilot" during live video interviews, shifting from static historical data to time-series streams:
* **Visual:** Gaze tracking (via MediaPipe) to detect sustained off-camera reading immediately post-question.
* **Audio:** Measuring millisecond latency between human question end and candidate response start to detect LLM processing delays.
* **Screen:** Tracking DOM events for application focus loss (tab-switching).
* **Action Space:** Instead of rejecting the candidate, the RL agent pings the human interviewer with real-time prompts (e.g., *"High latency detected. Ask candidate to whiteboard this concept."*).

---

## Usage & Installation

### Requirements
* Python 3.8+
* PyTorch
* Pandas
* Numpy
* Scikit-learn

