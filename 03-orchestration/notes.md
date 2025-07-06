# Orchestration

**orchestration** is about managing and coordinating the many interdependent steps that make up a full pipeline. 

During training or production pipelines, you often deal with multiple tasks:

* Data retrieval (e.g., pulling from APIs, databases, or cloud storage)
* Preprocessing (e.g., cleaning, transforming)
* Model training
* Hyperparameter tuning
* Logging, monitoring, alerting
* Retraining pipelines

Each of these steps can have:

* Dependencies (e.g., model can't train before data is ready)
* Failure risks (e.g., API downtime, corrupt data)
* Scalability concerns (e.g., parallelization)
* Maintenance overhead (e.g., retries, notifications)

So, you don't just need to *code the ML logic*. You also need to manage **how and when it runs**, **handle failures**, and **observe** what’s going on. That orchestration is crucial for reliability and automation.

Orchestration tools like **Prefect** come to the rescue and help in reducing such **Negative Engineering**.

### Negative Engineering

Negative engineering is the work that doesn't directly contribute to your end goal (e.g., training the model), but is necessary to make the system *resilient, reliable, and observable*.

Common forms:

* Retrying failed jobs (e.g., due to API issues)
* Handling malformed or unexpected data
* Sending notifications when things fail
* Adding logs and metrics to debug issues
* Writing conditional logic for partial failures
* Handling timeouts or resource limits

These tasks are **time-consuming**, **repetitive**, and **error-prone** if done manually, and they steal focus from core ML work.

### Solution: Tools like Prefect

**Prefect** is a modern workflow orchestration tool that reduces negative engineering by giving you infrastructure-level capabilities and automating many aspects of pipeline management:

* Retry logic out-of-the-box
* Failure notifications
* Conditional logic (what to do if step A fails)
* Timeouts and schedules
* Observability via dashboards
* Parallel and sequential execution
* Dependency tracking

It lets you focus on your business logic while handling orchestration reliably.

![](https://raw.githubusercontent.com/BPrasad123/MLOps_Zoomcamp/main/Week3/img/mlfowandprefect3.png)

<br>

Some of core concepts in Prefect:

- Tasks - They are functions having special runs when they should run; optionally take inputs, perform some work and optionally return an output.
- Workflows - They are basically containers of tasks defining dependencies among them.
- Modularity - Every component of Prefect is a modular design, making it easy to customize, to logging, to storage, to handle state.
- Concurrency - Supports massive concurrency
- Automation - Has a solid framework to support workflows

For more information about Prefect, visit [this link](https://docs.prefect.io/v3/get-started/quickstart).