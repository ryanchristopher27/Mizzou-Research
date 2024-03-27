# Mizzou-Research
Mizzou Master's Research


### Upstream to HPDI Repo
 - ```hpdi-upstream```

 

### Kubectl Commands
 - Copy directory to large pvc
    ```console
    cd /Mizzou-Research/research_container/container/src
    ```
    MMPretrain Configs
    ```
    kubectl cp mmpretrain_configs rc-large-pvc-pod:/rchristopher/data/src/
    ```
    Code
    ```
    kubectl cp code rc-large-pvc-pod:/rchristopher/data/src/
    ```

 - Copy from large pvc to directory
   ```console
   kubectl cp rc-large-pvc-pod:/rchristopher/data/src/mmpretrain_results/ucmerced_landuse ./mmpretrain_results/ucmerced_landuse
   ```

 - PVC Terminal
    ```console
    kubectl exec -it rc-large-pvc-pod -- /bin/bash
    ```

- Analysis Took
    ```console
    cd /Mizzou-Research/research_container/container/src/mmpretrain
    ```
    E.g.
    ```
    python tools/analysis_tools/analyze_logs.py plot_curve ./../results/mmpretrain_results/20240315_210743/vis_data/20240315_210743.json --keys loss --legend loss
    ```
