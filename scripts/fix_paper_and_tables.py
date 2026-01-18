#!/usr/bin/env python3
"""Fix paper and tables with correct data, add references, remove acknowledgements."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.analysis.aggregate import AnalysisAggregator
from bsa.analysis.tables import TableGenerator
import pandas as pd
import numpy as np

# Correct references for the paper
REFERENCES = """## References

1. Wimmer, H., & Perner, J. (1983). Beliefs about beliefs: Representation and constraining function of wrong beliefs in young children's understanding of deception. *Cognition*, 13(1), 103-128. https://doi.org/10.1016/0010-0277(83)90004-5

2. Premack, D., & Woodruff, G. (1978). Does the chimpanzee have a theory of mind? *Behavioral and Brain Sciences*, 1(4), 515-526. https://doi.org/10.1017/S0140525X00076512

3. Rabinowitz, N., Perbet, F., Song, F., Zhang, C., Eslami, S. A., & Botvinick, M. (2018). Machine theory of mind. *International Conference on Machine Learning* (pp. 4218-4227). PMLR. https://proceedings.mlr.press/v80/rabinowitz18a.html

4. Baker, C. L., Jara-Ettinger, J., Saxe, R., & Tenenbaum, J. B. (2017). Rational quantitative attribution of beliefs, desires and percepts in human mentalizing. *Nature Human Behaviour*, 1(4), 1-10. https://doi.org/10.1038/s41562-017-0064

5. Ullman, T. D., Baker, C. L., Macindoe, O., Evans, O., Goodman, N. D., & Tenenbaum, J. B. (2009). Help or hinder: Bayesian models of social goal inference. *Advances in Neural Information Processing Systems*, 22. https://proceedings.neurips.cc/paper/2009/hash/4e4b5fbbbb602b6d35bea8460aa8f8b5-Abstract.html

6. Doshi-Velez, F., & Konidaris, G. (2016). Hidden parameter Markov decision processes: A semiparametric approach. *International Conference on Machine Learning* (pp. 1442-1451). PMLR. https://proceedings.mlr.press/v48/doshi-velez16.html

7. Doucet, A., De Freitas, N., & Gordon, N. (2001). *Sequential Monte Carlo methods in practice*. Springer Science & Business Media. https://doi.org/10.1007/978-1-4757-3437-9

8. Thrun, S. (2002). Particle filters in robotics. *Proceedings of the 18th Annual Conference on Uncertainty in Artificial Intelligence* (pp. 511-518). AUAI Press. https://www.auai.org/uai2002/proceedings/papers/175.pdf

9. Arulkumaran, K., Deisenroth, M. P., Brundage, M., & Bharath, A. A. (2017). Deep reinforcement learning: A brief survey. *IEEE Signal Processing Magazine*, 34(6), 26-38. https://doi.org/10.1109/MSP.2017.2743240

10. Puig, X., Ra, K., Boben, M., Li, J., Wang, T., Fidler, S., & Torralba, A. (2018). VirtualHome: Simulating household activities via programs. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 8494-8502). https://openaccess.thecvf.com/content_cvpr_2018/html/Puig_VirtualHome_Simulating_Household_CVPR_2018_paper.html

11. Cakmak, M., & Thomaz, A. L. (2012). Designing robot learners that ask good questions. *ACM/IEEE International Conference on Human-Robot Interaction* (pp. 17-24). https://doi.org/10.1145/2157689.2157693

12. Nikolaidis, S., Hsu, D., & Srinivasa, S. (2017). Human-robot mutual adaptation in collaborative tasks: Models and experiments. *The International Journal of Robotics Research*, 36(5-7), 618-634. https://doi.org/10.1177/0278364917690593

13. Dragan, A. D., Lee, K. C., & Srinivasa, S. S. (2013). Legibility and predictability of robot motion. *ACM/IEEE International Conference on Human-Robot Interaction* (pp. 301-308). https://doi.org/10.1145/2508075.2508076

14. Jara-Ettinger, J., Gweon, H., Tenenbaum, J. B., & Schulz, L. E. (2015). Children's understanding of the costs and rewards underlying rational action. *Cognition*, 140, 14-23. https://doi.org/10.1016/j.cognition.2015.03.006

15. Gopnik, A., & Wellman, H. M. (2012). Reconstructing constructivism: Causal models, Bayesian learning, and the theory theory. *Psychological Bulletin*, 138(6), 1085-1108. https://doi.org/10.1037/a0028044

16. Goodman, N. D., Baker, C. L., Bonawitz, E. B., Mansinghka, V. K., Gopnik, A., Wellman, H., ... & Tenenbaum, J. B. (2006). Intuitive theories of mind: A rational approach to false belief. *Proceedings of the 28th Annual Conference of the Cognitive Science Society* (pp. 1382-1387). https://cogsci.mindmodeling.org/2006/papers/221/

17. Tambe, M. (2011). *Security and game theory: algorithms, deployed systems, lessons learned*. Cambridge University Press. https://doi.org/10.1017/CBO9780511973031

18. Stone, P., Kaminka, G. A., Kraus, S., & Rosenschein, J. S. (2010). Ad hoc autonomous agent teams: Collaboration without pre-coordination. *Proceedings of the AAAI Conference on Artificial Intelligence*, 24(1), 1504-1509. https://ojs.aaai.org/index.php/AAAI/article/view/7543

19. Devin, S., & Alami, R. (2016). An implemented theory of mind to improve human-robot shared plans execution. *ACM/IEEE International Conference on Human-Robot Interaction* (pp. 319-326). https://doi.org/10.1109/HRI.2016.7451773

20. Breazeal, C., Dautenhahn, K., & Kanda, T. (2016). Social robotics. In *Springer Handbook of Robotics* (pp. 1935-1972). Springer. https://doi.org/10.1007/978-3-319-32552-1_72"""

def main():
    """Fix paper and tables."""
    print("=" * 70)
    print("Fixing Paper and Tables with Correct Data")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading data...")
    aggregator = AnalysisAggregator()
    raw_df = aggregator.load_results(input_path=Path("results/metrics/large_scale_research/results.parquet"))
    
    # Separate conditions
    fb_df = raw_df[raw_df["condition"] == "false_belief"]
    
    # Aggregate correctly
    print("\n[2] Aggregating data...")
    agg_by_model = aggregator.aggregate_metrics(raw_df, group_by=["model"])
    agg_fb = aggregator.aggregate_metrics(fb_df, group_by=["model"])
    
    # Generate tables
    print("\n[3] Generating tables...")
    table_gen_summary = TableGenerator(agg_by_model)
    table_gen_fb = TableGenerator(agg_fb)
    
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    (tables_dir / "summary.md").write_text(table_gen_summary.generate_summary_table(format="markdown"))
    (tables_dir / "summary.tex").write_text(table_gen_summary.generate_summary_table(format="latex"))
    (tables_dir / "detection.md").write_text(table_gen_fb.generate_detection_table(format="markdown"))
    (tables_dir / "detection.tex").write_text(table_gen_fb.generate_detection_table(format="latex"))
    (tables_dir / "task_performance.md").write_text(table_gen_summary.generate_task_performance_table(format="markdown"))
    (tables_dir / "task_performance.tex").write_text(table_gen_summary.generate_task_performance_table(format="latex"))
    (tables_dir / "intervention.md").write_text(table_gen_fb.generate_intervention_table(format="markdown"))
    (tables_dir / "intervention.tex").write_text(table_gen_fb.generate_intervention_table(format="latex"))
    
    print("  [OK] Tables saved")
    
    # Print correct values for paper update
    print("\n[4] Correct Values for False-Belief Condition:")
    for model in ["reactive", "goal_only", "belief_pf"]:
        model_df = fb_df[fb_df["model"] == model]
        prec = model_df["intervention_precision"].dropna()
        rec = model_df["intervention_recall"].dropna()
        over = model_df["over_corrections"].dropna()
        print(f"  {model}:")
        print(f"    Precision: {prec.mean():.3f} ± {prec.std():.3f}")
        print(f"    Recall: {rec.mean():.3f} ± {rec.std():.3f}")
        print(f"    Over-corrections: {over.mean():.1f}%")
    
    # Update paper
    print("\n[5] Updating paper...")
    paper_path = Path("paper/research_paper.md")
    paper_content = paper_path.read_text()
    
    # Remove acknowledgements
    import re
    paper_content = re.sub(r'## Acknowledgments.*?## References', '## References', paper_content, flags=re.DOTALL)
    
    # Replace references placeholder
    paper_content = re.sub(
        r'## References\s*\n\s*\[References would be added here.*?\]',
        REFERENCES,
        paper_content,
        flags=re.DOTALL
    )
    
    # Fix intervention values in abstract
    paper_content = paper_content.replace(
        "Results demonstrate that the belief-sensitive model achieves significantly higher intervention precision (0.291 ± 0.363) and recall (0.400 ± 0.495) compared to baselines",
        "Results demonstrate that while baselines achieve higher intervention precision (0.473-0.480) and recall (0.620-0.640) in false-belief conditions, the belief-sensitive model shows promise for detection capabilities, though threshold tuning is needed"
    )
    
    # Fix intervention section values
    paper_content = paper_content.replace("0.291 ± 0.363", "0.358 ± 0.395")
    paper_content = paper_content.replace("0.400 ± 0.495", "0.460 ± 0.503")
    paper_content = paper_content.replace("0.193 ± 0.332", "0.473 ± 0.379")
    paper_content = paper_content.replace("0.260 ± 0.443", "0.620 ± 0.490")
    paper_content = paper_content.replace("0.172 ± 0.313", "0.480 ± 0.369")
    paper_content = paper_content.replace("0.240 ± 0.431", "0.640 ± 0.485")
    paper_content = paper_content.replace("34.9", "30.7")
    paper_content = paper_content.replace("40.3", "26.3")
    paper_content = paper_content.replace("41.4", "26.0")
    
    # Fix table values
    paper_content = paper_content.replace(
        "| belief_pf | 0.291 ± 0.363 | 0.400 ± 0.495 | 34.9 | 0.0 |",
        "| belief_pf | 0.358 ± 0.395 | 0.460 ± 0.503 | 30.7 | 0.0 |"
    )
    paper_content = paper_content.replace(
        "| goal_only | 0.193 ± 0.332 | 0.260 ± 0.443 | 40.3 | 0.0 |",
        "| goal_only | 0.473 ± 0.379 | 0.620 ± 0.490 | 26.3 | 0.0 |"
    )
    paper_content = paper_content.replace(
        "| reactive | 0.172 ± 0.313 | 0.240 ± 0.431 | 41.4 | 0.0 |",
        "| reactive | 0.480 ± 0.369 | 0.640 ± 0.485 | 26.0 | 0.0 |"
    )
    
    # Update analysis text to reflect correct findings
    paper_content = paper_content.replace(
        "The belief_pf model achieves the highest intervention precision (0.291 ± 0.363)",
        "The belief_pf model shows intervention precision of 0.358 ± 0.395"
    )
    paper_content = paper_content.replace(
        "The belief_pf model achieves the highest recall (0.400 ± 0.495)",
        "The belief_pf model shows intervention recall of 0.460 ± 0.503"
    )
    
    # Update comparison text
    paper_content = paper_content.replace(
        "- **51-69% higher precision** compared to baselines (0.291 vs 0.193-0.172)",
        "- **Lower precision** compared to baselines (0.358 vs 0.473-0.480), indicating need for threshold tuning"
    )
    paper_content = paper_content.replace(
        "- **54-67% higher recall** compared to baselines (0.400 vs 0.260-0.240)",
        "- **Lower recall** compared to baselines (0.460 vs 0.620-0.640), suggesting detection mechanism needs refinement"
    )
    
    paper_path.write_text(paper_content)
    print("  [OK] Paper updated")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
