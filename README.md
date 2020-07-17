# Document Anchors
![Test](https://github.com/kluge-ai/docanchors/workflows/Test/badge.svg?branch=master)

Implementation of an algorithm for the explanation of document classifications.

## Example

Basic setup of the two components:

```python
# Prepare fake instance and model interface
instance = np.random.randint(0, 100, size=25)


def get_predict_fn():
    """Load model from disk or prepare API calls."""

    def fake_model_predict(x: np.ndarray) -> np.ndarray:
        return np.random.randint(0, 2, size=x.shape)

    return fake_model_predict


# Set up sampling component
sample_queue = Queue(maxsize=10000)

batch_sampler = MultiprocessingSampler(BatchSampler(instance=instance,
                                                    get_predict_fn=get_predict_fn,
                                                    sample_queue=sample_queue,
                                                    target=1))

# Set up search component
generator = Generator(strategies=[Grow(), Shift()])

objective = Coherence() + 2.0 * AbsoluteCover(target=5)

doc_anchor = DocumentAnchor(sample_queue=sample_queue,
                            generator=generator,
                            objective=objective)

if __name__ == "__main__":
    # Start sampler
    batch_sampler.start()

    # Start search
    anchor = doc_anchor.explain(instance)

    # Terminate sampler
    batch_sampler.terminate()

    print(anchor)
```

For the full script, see [example.py](/example.py).

## Notes on design considerations

- [Creating a human-friendly interface for an XAI algorithm](https://www.kluge.ai/post/202007-human-friendly-interface/)

## Publications
- Kilian Kluge, Regina Eckhardt: *Explaining Suspected Phishing Attempts with Document Anchors.* [2020 ICML Workshop on Human Interpretability in Machine Learning](https://sites.google.com/view/whi2020/home), 2020

## References
- Kaufmann, E. and Kalyanakrishnan, S.: Information Complexity in Bandit Subset Selection. In: *Proceedings of
the 26th Annual Conference on Learning Theory*, pp. 228–251, 2013. [(online)](http://proceedings.mlr.press/v30/Kaufmann13.html)
- Lei, T., Barzilay, R., and Jaakkola, T.: Rationalizing Neural Predictions. In: *Proceedings of the 2016 Conference
on Empirical Methods in Natural Language Processing*, pp. 107–117. Association for Computational Linguistics, 2016. [doi:10.18653/v1/D16-1011](https://doi.org/10.18653/v1/D16-1011)
- Ribeiro, M., Singh, S., and Guestrin, C.: Anchors: High-Precision Model-Agnostic Explanations. In: *The Thirty-Second AAAI Conference on Artificial Intelligence*, pp. 1527–1535. Association for the
Advancement of Artificial Intelligence, 2018. [(online)](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16982/15850)

## License
Licensed under Apache 2. For more information, see [LICENSE](/LICENSE).
