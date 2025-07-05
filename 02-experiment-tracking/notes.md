# What is experiment tracking? 

It's the process of tracking all the relevant information of machine learning experiments, such as Code, Version, Environment, Data, Model, Artifacts, Metrics, Hyperparameters, etc.

This helps in reproducibility, better organized way of doing the projects.

## how do we track then? 

Well, at a very basic level one might use excel to track the information manually but that is error prone, difficult to collaborate and does not have a standard template to cater to all the needs of tracking mechanism.

That's why we'll use MLFlow for that purpose. 

---

We can launch mlflow ui as well. Run the following command to start mlflow ui (a gunicorn server) connected to the backend sqlite database.  
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## MLflow in Practice 
Depending upon the project and number of data scientists going to collaborate, the configurational aspect of mlflow is decided. Consider the following three scenarios.

- #### Scenario 1: A single data scientist participating in an ML competition

    MLflow setup:
    * Tracking server: no
    * Backend store: local filesystem
    * Artifacts store: local filesystem

    The experiments can be explored locally by launching the MLflow UI.

- #### Scenario 2: A cross-functional team with one data scientist working on an ML model

    MLflow setup:
    - tracking server: yes, local server
        -	Run:
            ```bash
            cd .\02-experiment-tracking\mlflow-scenarios\
            
            mlflow server --backend-store-uri sqlite:///backend.db --default-artifact-root ./artifacts_local
            ```
    - backend store: sqlite database
    - artifacts store: local filesystem

    The experiments can be explored locally by accessing the local tracking server.

    ---

    `mlflow ui`: Just a **local viewer** (communicates with local files or the backend DB directly, not via HTTP APIs). No REST API = no remote logging.

    `mlflow server`: Full **tracking server**, **with REST API** (suitable for real ML workflows), letting external tools/scripts do things like:

    - `POST /api/2.0/mlflow/runs/log-metric` → Log metrics

    - `GET /api/2.0/mlflow/experiments/list` → List experiments

    - `DELETE /api/2.0/mlflow/runs/delete` → Delete a run

    - etc.

- #### Scenario 3: Multiple data scientists working on multiple ML models

    MLflow setup:
    * Tracking server: yes, remote server (EC2).
    * Backend store: postgresql database.
    * Artifacts store: s3 bucket.

    The experiments can be explored by accessing the remote server.

<br>

### Model selection 

To decide on a model to put into a production environment, you should look at:

- The loss metric

- The time it took to train (models that took longer to train, are usually more complex)

- The time it takes to test 

- The size of the model

## MLflow: benefits, limitations and alternatives 

**Benefits**
* Share and collaborate with other members
* More visibility into all the efforts
  
**Limitations**
* Security - restricting access to the server
* Scalability
* Isolation - restricting access to certain artifacts

**When not to use**
* Authentication and user profiling is required
* Data versioning - no in-built functionality but there are work arounds
* Real-time model/data monitoring and alerts are required (e.g., detecting data drift, accuracy decay, or raising alerts on failures)

**Alternates**
* Nepture.ai
* Comet.ai
* Weights and Biases
* etc