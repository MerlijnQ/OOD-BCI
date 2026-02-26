class hyperparam():
    """
    A class to provide predefined hyperparameters for KNN and DUQ
    models based on the dataset used in the experiments.

    This class centralizes hyperparameter management, allowing
    consistent access to the parameters required for OOD detection
    experiments.
    """
    def get_hyperparameters(self,
                            dataset_name: str
                            ) -> dict:
        """
        Return and store hyperparameters for KNN and DUQ for a given dataset.

        Args:
            dataset_name (str): The name of the dataset for which
                                hyperparameters are requested.

        Raises:
            NotImplementedError: If no hyperparameters are defined for
                                 the specified dataset.

        Returns:
            dict: A dictionary containing hyperparameter values for KNN and
                DUQ. Keys are dataset-specific label names.
                For inversion experiments, this returns a nested dictionary
                with label-specific parameters.
        """
        match dataset_name:
            case "Schirrmeister2017":
                values = {
                    "right_hand": {
                        "k": 18,
                        "centroid_size": 32,
                        "penalty_w": 0.00013346977574178097,
                        "one_minus_gamma": 0.01120760621186057
                    },
                    "left_hand": {
                        "k": 4,
                        "centroid_size": 32,
                        "penalty_w": 0.00013346977574178097,
                        "one_minus_gamma": 0.01120760621186057
                    },
                    "rest": {
                        "k": 85,
                        "centroid_size": 256,
                        "penalty_w": 0.037311604524743164,
                        "one_minus_gamma": 0.041380401125610165
                    },
                    "feet": {
                        "k": 62,
                        "centroid_size": 64,
                        "penalty_w": 3.776663327107335e-05,
                        "one_minus_gamma": 0.002051110418843397
                    }
                }
            case "Schirrmeister2017_inverse":
                values = {
                    "right_hand": {
                        "left_hand": {
                            "k": 50,
                            "centroid_size": 64,
                            "penalty_w": 0.001553544580758846,
                            "one_minus_gamma": 0.001238513729886093
                        },
                        "rest": {
                            "k": 7,
                            "centroid_size": 64,
                            "penalty_w": 3.776663327107335e-05,
                            "one_minus_gamma": 0.002051110418843397
                        },
                        "feet": {
                            "k": 1,
                            "centroid_size": 128,
                            "penalty_w": 0.0002471471974904756,
                            "one_minus_gamma": 0.0030858626026361636
                        }
                    },
                    "left_hand": {
                        "right_hand": {
                            "k": 71,
                            "centroid_size": 64,
                            "penalty_w": 0.001553544580758846,
                            "one_minus_gamma": 0.001238513729886093
                        },
                        "rest": {
                            "k": 1,
                            "centroid_size": 32,
                            "penalty_w": 0.00013346977574178097,
                            "one_minus_gamma": 0.01120760621186057
                        },
                        "feet": {
                            "k": 1,
                            "centroid_size": 64,
                            "penalty_w": 3.776663327107335e-05,
                            "one_minus_gamma": 0.002051110418843397
                        }
                    },
                    "rest": {
                        "right_hand": {
                            "k": 45,
                            "centroid_size": 128,
                            "penalty_w": 0.00012040216379191709,
                            "one_minus_gamma": 0.005404103854647328
                        },
                        "left_hand": {
                            "k": 71,
                            "centroid_size": 32,
                            "penalty_w": 0.002091173544376496,
                            "one_minus_gamma": 0.024673775376624445
                        },
                        "feet": {
                            "k": 40,
                            "centroid_size": 64,
                            "penalty_w": 1.1916299962955139e-05,
                            "one_minus_gamma": 0.08706020878304858
                        }
                    },
                    "feet": {
                        "right_hand": {
                            "k": 3,
                            "centroid_size": 64,
                            "penalty_w": 3.776663327107335e-05,
                            "one_minus_gamma": 0.002051110418843397
                        },
                        "left_hand": {
                            "k": 54,
                            "centroid_size": 64,
                            "penalty_w": 3.776663327107335e-05,
                            "one_minus_gamma": 0.002051110418843397
                        },
                        "rest": {
                            "k": 84,
                            "centroid_size": 128,
                            "penalty_w": 0.00012040216379191709,
                            "one_minus_gamma": 0.005404103854647328
                        }
                    }
                }
            case "BNCI2014_001":
                values = {
                    "left_hand": {
                        "k": 1,
                        "centroid_size": 64,
                        "penalty_w": 0.001553544580758846,
                        "one_minus_gamma": 0.001238513729886093
                    },
                    "right_hand": {
                        "k": 2,
                        "centroid_size": 128,
                        "penalty_w": 0.00012040216379191709,
                        "one_minus_gamma": 0.005404103854647328
                        },
                    "feet": {
                        "k": 12,
                        "centroid_size": 64,
                        "penalty_w": 1.1916299962955139e-05,
                        "one_minus_gamma": 0.08706020878304858
                    },
                    "tongue": {
                            "k": 106,
                            "centroid_size": 128,
                            "penalty_w": 0.00012040216379191709,
                            "one_minus_gamma": 0.005404103854647328
                        }
                    }
            case "BNCI2014_001_inverse":
                values = {
                    "left_hand": {
                        "right_hand": {
                            "k": 92,
                            "centroid_size": 64,
                            "penalty_w": 3.776663327107335e-05,
                            "one_minus_gamma": 0.002051110418843397
                        },
                        "feet": {
                            "k": 2,
                            "centroid_size": 32,
                            "penalty_w": 0.00013346977574178097,
                            "one_minus_gamma": 0.01120760621186057
                        },
                        "tongue": {
                            "k": 5,
                            "centroid_size": 32,
                            "penalty_w": 0.00013346977574178097,
                            "one_minus_gamma": 0.01120760621186057
                        }
                    },
                    "right_hand": {
                        "left_hand": {
                            "k": 92,
                            "centroid_size": 32,
                            "penalty_w": 0.00013346977574178097,
                            "one_minus_gamma": 0.01120760621186057
                        },
                        "feet": {
                            "k": 87,
                            "centroid_size": 32,
                            "penalty_w": 0.0005673857029576254,
                            "one_minus_gamma": 0.015253421517969643
                        },
                        "tongue": {
                            "k": 5,
                            "centroid_size": 64,
                            "penalty_w": 1.1916299962955139e-05,
                            "one_minus_gamma": 0.08706020878304858
                        }
                    },
                    "feet": {
                        "left_hand": {
                            "k": 48,
                            "centroid_size": 64,
                            "penalty_w": 0.001553544580758846,
                            "one_minus_gamma": 0.001238513729886093
                        },
                        "right_hand": {
                            "k": 7,
                            "centroid_size": 64,
                            "penalty_w": 1.1916299962955139e-05,
                            "one_minus_gamma": 0.08706020878304858
                        },
                        "tongue": {
                            "k": 77,
                            "centroid_size": 128,
                            "penalty_w": 0.0037748871070193107,
                            "one_minus_gamma": 0.021404181880240392
                        }
                    },
                    "tongue": {
                        "left_hand": {
                            "k": 92,
                            "centroid_size": 128,
                            "penalty_w": 0.00012040216379191709,
                            "one_minus_gamma": 0.005404103854647328
                        },
                        "right_hand": {
                            "k": 92,
                            "centroid_size": 64,
                            "penalty_w": 3.776663327107335e-05,
                            "one_minus_gamma": 0.002051110418843397
                        },
                        "feet": {
                            "k": 92,
                            "centroid_size": 256,
                            "penalty_w": 0.037311604524743164,
                            "one_minus_gamma": 0.041380401125610165
                        }
                    }
                }
            case "Stieger2021":
                values = {
                    "right_hand": {
                        "k": 9,
                        "centroid_size": 256,
                        "penalty_w": 0.0005666713911103605,
                        "one_minus_gamma": 0.09841513366747653
                    },
                    "left_hand": {
                        "k": 43,
                        "centroid_size": 64,
                        "penalty_w": 0.0001422361277374357,
                        "one_minus_gamma": 0.010968217207529524
                    },
                    "both_hand": {
                        "k": 2,
                        "centroid_size": 32,
                        "penalty_w": 0.011141678989786418,
                        "one_minus_gamma": 0.005030582354098115
                    },
                    "rest": {
                        "k": 303,
                        "centroid_size": 32,
                        "penalty_w": 0.00013346977574178097,
                        "one_minus_gamma": 0.01120760621186057
                    }
                }
            case "Stieger2021_inverse":
                values = {
                    "right_hand": {
                        "left_hand": {
                            "k": 46,
                            "centroid_size": 32,
                            "penalty_w": 0.02138374822441,
                            "one_minus_gamma": 0.03036032262967418
                            },
                        "both_hand": {
                            "k": 183,
                            "centroid_size": 32,
                            "penalty_w": 0.0015133596244653014,
                            "one_minus_gamma": 0.02977933806625081
                            },
                        "rest": {
                            "k": 4,
                            "centroid_size": 128,
                            "penalty_w": 0.0029019866532834326,
                            "one_minus_gamma": 0.014927222084177889
                            }
                    },
                    "left_hand": {
                        "right_hand": {
                            "k": 34,
                            "centroid_size": 128,
                            "penalty_w": 0.00032810231411814386,
                            "one_minus_gamma": 0.0160750838129177
                            },
                        "both_hand": {
                            "k": 3,
                            "centroid_size": 64,
                            "penalty_w": 3.776663327107335e-05,
                            "one_minus_gamma": 0.002051110418843397
                            },
                        "rest": {
                            "k": 44,
                            "centroid_size": 128,
                            "penalty_w": 0.00022631620724633536,
                            "one_minus_gamma": 0.050412209949909956
                            }
                    },
                    "both_hand": {
                        "right_hand": {
                            "k": 1,
                            "centroid_size": 128,
                            "penalty_w": 0.00026508324836235536,
                            "one_minus_gamma": 0.013587341105396984
                            },
                        "left_hand": {
                            "k": 11,
                            "centroid_size": 128,
                            "penalty_w": 0.011897650159891733,
                            "one_minus_gamma": 0.017311542042698063
                            },
                        "rest": {
                            "k": 97,
                            "centroid_size": 256,
                            "penalty_w": 0.005141549905871665,
                            "one_minus_gamma": 0.09535264289457136
                            }
                    },
                    "rest": {
                        "right_hand": {
                            "k": 274,
                            "centroid_size": 32,
                            "penalty_w": 0.0034260482540377263,
                            "one_minus_gamma": 0.0037008002789962807
                            },
                        "left_hand": {
                            "k": 27,
                            "centroid_size": 32,
                            "penalty_w": 0.0020244991733450704,
                            "one_minus_gamma": 0.058589591995037674
                            },
                        "both_hand": {
                            "k": 145,
                            "centroid_size": 128,
                            "penalty_w": 0.029866092370009447,
                            "one_minus_gamma": 0.06161049539380966
                            }
                    }
                }
            case _:
                raise NotImplementedError(
                    f"Hyperparameters for {dataset_name} not implemented yet")
        return values
