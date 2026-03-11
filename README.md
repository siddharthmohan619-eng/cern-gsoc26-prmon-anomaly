CERN-HSF GSoC 2026: Automated Software Performance Monitoring

1. Introduction & Setup

This repository contains the warm-up exercise for the ATLAS anomaly detection project. Because prmon relies on the Linux /proc filesystem, I ran the data generation phase inside an isolated Ubuntu 22.04 Docker container on my macOS host to ensure accurate metric collection.

Dependencies used: prmon (compiled from source), C++, Bash, Python 3 (pandas, scikit-learn, matplotlib).



2. Data Generation & Anomaly Injection

I used the built-in prmon-burner test to generate time-series data. I created a bash script (run_test.sh) to alternate between a stable baseline and artificial anomalies.

The Test Parameters:



Baseline: 2 threads, 500MB memory allocation.

Anomaly 1 (Memory Spike): 2 threads, 2048MB memory allocation.

Anomaly 2 (Thread Spike): 16 threads, 500MB memory allocation.

Execution Command:



Bash



# Executing prmon to monitor the burner script at 1-second intervals

./prmon --interval 1 -- ./run_test.sh

3. Anomaly Detection Approach

For detection, I used an Isolation Forest machine learning model via scikit-learn.

Why Isolation Forest?



It is an unsupervised algorithm, which is ideal here since we do not have a pre-labeled "normal" vs "abnormal" dataset in real-world scenarios.

It handles multivariate data well. By feeding it both pss (Memory) and nprocs (Threads), it can identify anomalies that occur across different resource dimensions simultaneously.

Code Snippet (Detection Logic):



Python



# Extracting features and training the model

features = ['pss', 'nprocs']

model = IsolationForest(contamination=0.15, random_state=42)

df['anomaly'] = model.fit_predict(df[features]) # Outputs -1 for anomalies

4. Results & Visualization

The model successfully flagged both the memory spike and the thread spike.



5. Conclusions & Trade-offs

The Isolation Forest effectively identified the injected anomalies without requiring manual thresholding. However, there are trade-offs:



Pros: Highly robust to multi-dimensional anomalies; automatically adapts to the shape of the data without hardcoded limits.

Cons: Requires tuning the contamination parameter (the expected proportion of outliers). For real-time, lightweight monitoring of a single metric (like just memory), a simple rolling Z-score statistical method might be more computationally efficient than a tree-based ML model.

AI Disclosure: I used Gemini to assist with navigating Docker container file paths on macOS, structuring the bash data generation script, formatting the scikit-learn Isolation Forest implementation, and refining the narrative structure of this README. All design decisions and code execution were managed by me i tried my best.
