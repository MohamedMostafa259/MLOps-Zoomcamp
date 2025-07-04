**MLOps** = Set of best practices for putting ML into production

## ML Project Stages

We can think of 3 stages in an ML project:

- **Design:** Is ML really the right solution?

- **Train:** Find the best possible model

- **Operate:** Run/deploy the model, also evaluate and update the model


## Model Maturity

Based on a [Microsoft Article](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model), there are roughly 5 levels of maturity ranging from 0 (no automation) through to 4 (fully automated).

- **0: No MLOps**
    - All code is in sloppy notebooks. This is sufficient for PoC (Proof of Concept).

- **1: DevOps but no MLOps**
    - Releases are automated,
    - Unit and integrations testing
    - CI/CD pipelines set up for re-deployment
    - OPS metrics (tracking request amounts, and other infrastructure/software metrics) 
    
    It does not
    - Track experiments
    - Ensure reproducibility

    In short data scientists are separated from the engineers. Often this is the stage when the models are getting ready to go into production.

- **2: Automated training**

    -   The training pipeline is automated
    -   Experiment tracking
    -   Model registry (Usually if more than 2-3 models are being used)
    -   Low friction deployment

    Often data scientists work with engineers

- **3: Automated deployment**

    -   Easy to deploy a model (e.g. through an API call)
    -   Data prep, training, and deployment are all done in one pipeline
    -   A/B testing
    -   Comparing models to see which performs better
    -   Models are monitored as part of the deployment process

- **4: Full MLOps automation**

    -   Everything happens automatically

<br>

Often you do not really need to get to the higher stages such as levels 3 and 4. In many cases, level 2 is sufficient for most software products.

This course will go up to levels 2/3.