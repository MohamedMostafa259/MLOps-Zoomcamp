This module focuses on the **operate phase** of the MLOps lifecycle, where we take trained models and deploy them for production use. 

### Deployment Decision Framework

The key question when choosing a deployment strategy is: **"Do you need predictions immediately, or can you wait?"**

- Can wait (minutes/hours/days) → **Batch Deployment**
- Need immediate results → **Online Deployment**
  - Prediction per item (e.g., user, transaction: send one request, get one response) → **Web Service**
  - Monitoring, real-time analytics, pattern detection on live data (continuous flow of data) → **Streaming**

![](https://raw.githubusercontent.com/BPrasad123/MLOps_Zoomcamp/main/Week4/img/predwebservice_v1.png)


### Typical Batch Architecture

```
[Database / Data Lake] 
        ↓ (Scheduled)
  [Scoring Job / Model Inference]
        ↓
[Predictions Stored in DB / CSV / Warehouse]
        ↓
[Reporting Dashboards / Business Actions / Notifications]
```


### Typical Web Service Architecture

```
[Client / Frontend / App / External System]
        ↓ (Sends request)
   [API Gateway or Load Balancer]
        ↓
     [Model Server (Flask, FastAPI, TorchServe, etc.)]
        ↓
     [Prediction Returned Instantly]
```


### Typical Streaming Architecture 

```
Producer → Event Stream → Multiple Consumers
```

- **Example:**
    ```
    Video Upload → Event Stream → [Copyright Detector, NSFW Detector, Violence Detector] → Decision Service
    ```

