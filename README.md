# Learning a Mini-batch Graph Transformer via Two-stage Interaction Augmentation

 Official codebase for paper [Learning a Mini-batch Graph Transformer via Two-stage Interaction Augmentation](). 


## 1. Prerequisites

#### Install dependencies

See `requirment.txt` file for more information about how to install the dependencies.


## 2. Run the code
We provide scripts to replicate the results in the paper.


```bash
sh run.sh
```

## Additional Information: Time Comparison
Table R1:
The running times for large-scale datasets were recorded. The reported times represent the model training time for a single epoch, measured in seconds.
| Methods       | ogbn-arxiv    | pokec     | twitch-gamer  |
|---------------|---------------|-----------|---------------|
| DIFFormer     | 0.403         | 4.121     | 0.595         |
| NodeFormer    | 0.989         | 12.827    | 1.189         |
| NAGphormer    | 1.857         | 17.460    | 1.798         |
| GOAT          | 13.523        | 628.67    | 92.55         |
| LGMformer     | 24.746        | 157.419   | 58.202        |

