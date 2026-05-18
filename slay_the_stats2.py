import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, brier_score_loss


#print(runs[3])

#df.pop("acts")
#print(df)
#print(df.describe())
#for key in df:
#    print(key)
#print(df["killed_by_encounter"].value_counts(normalize=True).head(5))
#print(df["win"].mean())
#df["ascension"].hist(bins=10)
#plt.title("Ascentions")
#plt.xlabel("Level")
#plt.ylabel("Num runs")
#plt.show()

#print(df["map_point_history"][0][0][0]["player_stats"][0]["current_hp"])

#print(df["map_point_history"])

class StatsInterface:
    def __init__(self, runs_folder:str="runs"):
        self.runs_folder = runs_folder
        self.close = False
        self.df = None
        self.overall_win_rate = None
        self.total_wins = 0
        self.total_runs = 0
        self.runs = dict()                  # FORMAT:
                                            # {run_number: [character_name, win_boolean, ascension_number,
                                            # number_of_rooms, total_damage_taken, average_damage_taken_each_floor,
                                            # killed_by_encounter, killed_by_event], ...}
        self.all_played_characters = dict() # FORMAT:
                                            # {"character":[total_runs, total_wins, win_rate], ...}
        self.total_damage_taken = 0
        self.average_damage_taken = 0

        self.damage_winning_runs = []
        self.damage_losing_runs = []
        self.damage_per_floor_winning_runs = []
        self.damage_per_floor_losing_runs = []
        self.all_runs_relics = dict()

        self.act1_relics_winning = []
        self.act1_relics_losing = []
        self.all_runs_and_relics_in_act1 = []
        self._last_train_mask = None
        # ekstra tall fra map_point_history  (gull, hvile, elites, osv) 
        self.run_gold_spent = []
        self.run_gold_gained = []
        self.run_hp_healed = []
        self.run_rest_sites = []
        self.run_shop_visits = []
        self.run_elite_fights = []
        self.run_boss_fights = []
        self.run_monster_rooms = []
        self.run_deck_size = []
    def start_inferface(self):
        if self.df is None:
            Exception("ERROR: Data not loaded.")

        print("Welcome to the Sts2 interface\n"
              "Below is a list of commands you can run in order to extract different statistics "
              "from your games:")

        while True:
            print("1. 'overall_win_rate'\n"
                  "2. 'characters' (for stats on your characters)\n"
                  "3. 'most_common_causes_of_death'\n"
                  "4. 'mean_damage_taken'\n"
                  "5. 'plot_floors_reached'\n"
                  "6. 'plot_win_rate_per_character'\n"
                  "7. 'average_damage_winning_vs_losing'\n"
                  "8. 'average_number_of_relics_act1_losing_vs_winning'\n"
                  "9. 'bootstrap_calculations'\n"
                  "10. 'linear_regression'\n"
                  "11. 'logistic_regression'\n"
                  "12. 'average_run_time_per_floor'\n"
                  "13. 'logistic_evaluation'\n"
                  "14. 'naive_bayes_compare'\n"
                  "15. 'permutation_test'\n"
                  "16. 'prediction_intervals'\n"
                  "17. 'play_smarter'\n"
                  "18. 'all'\n"
                  "19. 'q' - to quit")
            action = input("Command: ")
            if action == "q":
                followup = input("Are you certain you want to quit? (y/n): ")
                if followup.lower() == "y" or followup.lower == "yes":
                    print("\nQuitting...")
                    self.close = True
                    break
                else:
                    print("Quit aborted.\n")
            elif action == "overall_win_rate":
                self.print_overall_win_rate()
            elif action == "characters":
                self.print_characters()
            elif action == "most_common_causes_of_death":
                self.print_most_common_causes_of_death()
            elif action == "all":
                self.output_all()
            elif action == "plot_floors_reached":
                self.plot_floors_reached()
            elif action == "plot_win_rate_per_character":
                self.plot_win_rate_per_character()
            elif action == "average_damage_winning_vs_losing":
                self.print_average_damage_winning_vs_losing()
            elif action == "average_number_of_relics_act1_losing_vs_winning":
                self.print_average_number_of_relics_act1_losing_vs_winning()
            elif action == "bootstrap_calculations":
                abort = 0
                while True and abort < 3:
                    print("1. 'win_rate'\n"
                          "2. 'average_damage_taken_per_floor'\n"
                          "3. 'scipy_bootstrap' (same as 1 or 2 but with scipy.stats.bootstrap)")
                    choice = input("Which of the above do you want to bootstrap? (enter for win_rate) ")
                    if len(choice) == 0:
                        choice = "win_rate"
                        break
                    elif choice.lower() == "win_rate":
                        break
                    elif choice.lower() == "average_damage_taken_per_floor":
                        break
                    elif choice.lower() == "scipy_bootstrap":
                        break
                    else:
                        print(f"Please write a legal option. Remaining attempts: {2-abort}")
                        abort += 1
                        choice = "win_rate"

                use_scipy = choice.lower() == "scipy_bootstrap"
                if use_scipy:
                    abort = 0
                    while True and abort < 3:
                        print("1. 'win_rate'\n"
                              "2. 'average_damage_taken_per_floor'")
                        scipy_choice = input("Scipy bootstrap for which? (enter for win_rate) ")
                        if len(scipy_choice) == 0:
                            choice = "win_rate"
                            break
                        elif scipy_choice.lower() == "win_rate":
                            choice = "win_rate"
                            break
                        elif scipy_choice.lower() == "average_damage_taken_per_floor":
                            choice = "average_damage_taken_per_floor"
                            break
                        else:
                            print(f"Please write a legal option. Remaining attempts: {2-abort}")
                            abort += 1
                            choice = "win_rate"

                abort = 0
                while True and abort < 3:
                    num_bootstrap_samples = input("Num boostrap samples (enter for 10 000): ")
                    if len(num_bootstrap_samples) == 0:
                        num_bootstrap_samples = 10000
                        break
                    else:
                        try:
                            num_bootstrap_samples = int(num_bootstrap_samples)
                            if num_bootstrap_samples >= 1:
                                break
                            else:
                                print("Please write a legal number larger than 1 or press enter only.")
                                abort += 1
                        except ValueError:
                            print("Please write a legal number larger than 1 or press enter only.")
                            abort += 1

                if abort < 3:
                    if use_scipy:
                        if choice == "win_rate":
                            print("WIN RATE (scipy bootstrap):")
                            self.calculate_scipy_bootstrap_and_print(self.df["win"].to_list(), "Win rate", num_bootstrap_samples)
                        else:
                            print("DAMAGE TAKEN PER FLOOR ON WINNING RUNS (scipy bootstrap):")
                            self.calculate_scipy_bootstrap_and_print(self.damage_per_floor_winning_runs, "Damage taken per floor on winning runs", num_bootstrap_samples)
                            print("DAMAGE TAKEN PER FLOOR ON LOSING RUNS (scipy bootstrap):")
                            self.calculate_scipy_bootstrap_and_print(self.damage_per_floor_losing_runs, "Damage taken per floor on losing runs", num_bootstrap_samples)
                    elif choice == "win_rate":
                        print("WIN RATE:")
                        self.calculate_bootstrap_and_print(self.df["win"].to_list(), "Win rate", num_bootstrap_samples)
                    else:
                        print("DAMAGE TAKEN PER FLOOR ON WINNING RUNS:")
                        self.calculate_bootstrap_and_print(self.damage_per_floor_winning_runs, "Damage taken per floor on winning runs",num_bootstrap_samples)
                        print("DAMAGE TAKEN PER FLOOR ON LOSING RUNS:")
                        self.calculate_bootstrap_and_print(self.damage_per_floor_losing_runs, "Damage taken per floor on losing runs", num_bootstrap_samples)
            elif action == "linear_regression":
                self.linear_regression()
            elif action == "logistic_regression":
                self.logistic_regression()
            elif action == "average_run_time_per_floor":
                self.regression_calculations()
                print(f"Average run time per floor is {self.df["run_time_per_floor"].mean():.3f} seconds.")
            elif action == "logistic_evaluation":
                self.evaluate_logistic_on_test()
            elif action == "naive_bayes_compare":
                self.naive_bayes_and_compare()
            elif action == "permutation_test":
                self.permutation_test_damage()
            elif action == "prediction_intervals":
                self.prediction_intervals()
            elif action == "play_smarter":
                self.linear_regression_play_insights()
            else:
                print("Unknow command. Type 'q' to quit.")

    def print_overall_win_rate(self):
        print(f"Mean win rate: {self.overall_win_rate:.3f}\nE.g. ~ {self.overall_win_rate * 100:.1f}%")
        t_stat, p_value = stats.ttest_1samp(self.df["win"], 0.5)
        print(f"One-sample t test (H0: win rate = 0.5): T value {t_stat:.3f}, P value {p_value:.3f}\n")

    def print_characters(self):
        for character, value in self.all_played_characters.items():
            print(f"{character.lower().capitalize()}: {value[0]} total runs, {value[1]} total wins; {value[2]:.3f} in win rate.")

    def print_most_common_causes_of_death(self):
        print("ENCOUNTERS:")
        for encounter, count in self.df["killed_by_encounter"].value_counts().head(5).items():
            print(encounter.split(".")[1].lower().capitalize().strip(), ": ", count)
        print("EVENTS:")
        for event, count in self.df["killed_by_event"].value_counts().head(5).items():
            print(event.split(".")[1].lower().capitalize().strip(), ": ", count)

    def print_mean_damage_taken(self):
        print(f"Mean damage taken across all runs: {self.average_damage_taken:.3f}")

    def plot_floors_reached(self):
        floors = list()
        for run_number, run in self.runs.items():
            floors.append(run[3])

        plt.hist(floors, bins="auto")
        plt.xlabel("Number of floors")
        plt.ylabel("Number of floors reaching this range")
        plt.title("Floors reached histogram")
        plt.show()

    def plot_win_rate_per_character(self):
        characters = []
        win_rates = []
        for character, value in self.all_played_characters.items():
            characters.append(character)
            win_rates.append(value[2])
        plt.bar(characters, win_rates)
        plt.xlabel("Characters")
        plt.ylabel("Win rate")
        plt.title("Win rate per character")
        plt.show()

    def print_average_damage_winning_vs_losing(self):
        abort = 0
        while True and abort < 3:
            significance_treshold = input("Significance treshold (enter for 0.05): ")
            if len(significance_treshold) == 0:
                significance_treshold = 0.05
                break
            else:
                try:
                    if float(significance_treshold) > 1 or float(significance_treshold) < 0:
                        print("This is not an acceptable value, please type an legal number between 0 and 1.")
                        abort += 1
                    else:
                        significance_treshold = float(significance_treshold)
                        break
                except ValueError:
                    print("This is not an acceptable value, please type an legal number between 0 and 1.")
                    abort += 1
        t_stat, p_value = stats.ttest_ind(self.damage_winning_runs, self.damage_losing_runs, equal_var=False)
        print(f"Average total damage taken on winning runs: "
              f"{(sum(self.damage_winning_runs)/len(self.damage_winning_runs) if len(self.damage_winning_runs) != 0 else 0):.3f}\n"
              f"Median total damage on winning runs: "
              f"{(np.median(self.damage_winning_runs) if len(self.damage_winning_runs) != 0 else 0):.3f}\n"
              f"Average total damage taken on losing runs: "
              f"{(sum(self.damage_losing_runs)/len(self.damage_losing_runs) if len(self.damage_losing_runs) != 0 else 0):.3f}\n"
              f"Median total damage on losing runs: "
              f"{(np.median(self.damage_losing_runs) if len(self.damage_losing_runs) != 0 else 0):.3f}\n"
              f"T value: {t_stat:.3f} , P value {p_value:.3f} (Welch t test)\n"
              f"This is {f'' if p_value < significance_treshold else '_not_ '}statistically significant given significant treshold at {significance_treshold}.\n")

        t_stat, p_value = stats.ttest_ind(self.damage_per_floor_winning_runs, self.damage_per_floor_losing_runs, equal_var=False)
        print(f"Average per floor damage taken on winning runs: "
              f"{(sum(self.damage_per_floor_winning_runs) / len(self.damage_per_floor_winning_runs) if len(self.damage_per_floor_winning_runs) != 0 else 0):.3f}\n"
              f"Median per floor damage on winning runs: "
              f"{(np.median(self.damage_per_floor_winning_runs) if len(self.damage_per_floor_winning_runs) != 0 else 0):.3f}\n"
              f"{(sum(self.damage_per_floor_losing_runs) / len(self.damage_per_floor_losing_runs) if len(self.damage_per_floor_losing_runs) != 0 else 0):.3f}\n"
              f"Median per floor damage on losing runs: "
              f"{(np.median(self.damage_per_floor_losing_runs) if len(self.damage_per_floor_losing_runs) != 0 else 0):.3f}\n"
              f"T value: {t_stat:.3f} , P value {p_value:.3f} (Welch t test)\n"
              f"This is {f'' if p_value < significance_treshold else '_not_ '}statistically significant given significant treshold at {significance_treshold}.\n")

    def print_average_number_of_relics_act1_losing_vs_winning(self):
        abort = 0
        while True and abort < 3:
            significance_treshold = input("Significance treshold (enter for 0.05): ")
            if len(significance_treshold) == 0:
                significance_treshold = 0.05
                break
            else:
                try:
                    if float(significance_treshold) > 1 or float(significance_treshold) < 0:
                        print("This is not an acceptable value, please type an legal number between 0 and 1.")
                        abort += 1
                    else:
                        significance_treshold = float(significance_treshold)
                        break
                except ValueError:
                    print("This is not an acceptable value, please type an legal number between 0 and 1.")
                    abort += 1

        t_stat, p_value = stats.ttest_ind(self.act1_relics_winning, self.act1_relics_losing,
                                          equal_var=False)

        print(f"Average number of relics by end of act 1 on winning runs: "
              f"{(sum(self.act1_relics_winning) / len(self.act1_relics_winning) if len(self.act1_relics_winning) != 0 else 0):.3f}\n"
              f"Average number of relics by end of act 1 on losing runs: "
              f"{(sum(self.act1_relics_losing) / len(self.act1_relics_losing) if len(self.act1_relics_losing) != 0 else 0):.3f}\n"
              f"T value: {t_stat:.3f} , P value {p_value:.3f} (Welch t test)\n"
              f"This is {f'' if p_value < significance_treshold else '_not_ '}statistically significant given significant treshold at "
              f"{significance_treshold}.\n")

    def output_all(self):
        self.print_overall_win_rate(); print()
        self.print_characters(); print()
        self.print_most_common_causes_of_death(); print()
        self.print_mean_damage_taken(); print()
        self.plot_floors_reached(); print()
        self.plot_win_rate_per_character(); print()
        self.print_average_damage_winning_vs_losing(); print()
        self.print_average_number_of_relics_act1_losing_vs_winning(); print()

    def load_dataframe(self):
        # .run filer er json: win, ascension, run_time, players[{character, relics}], map_point_history[[rom...]]
        runs = []
        for filename in os.listdir(self.runs_folder):
            if filename.endswith(".run"):
                file_path = os.path.join(self.runs_folder, filename)  # spesifiserer path
                # bruker pandas til å lese,
                run = pd.read_json(file_path, typ="series")  # "series" fordi hvert run er ett enkelt objekt
                runs.append(run)
        df = pd.DataFrame(runs)
        df.pop("acts") # this is useless data
        self.df = df

    def calculate_bootstrap_and_print(self, observations:list, label:str, num_bootstrap_samples:int=10000):
        # BOOTSTRAPING
        np.random.seed(777)
        bootstrap_samples_mean_results = []
        observations = np.array(observations)
        for _ in range(num_bootstrap_samples):
            bootstrap_sample = np.random.choice(observations, self.total_runs, replace = True)
            bootstrap_sample_mean = sum(bootstrap_sample) / len(bootstrap_sample)
            bootstrap_samples_mean_results.append(bootstrap_sample_mean)
        bootstrap_mean = sum(bootstrap_samples_mean_results) / num_bootstrap_samples
        lower, upper = np.percentile(bootstrap_samples_mean_results, [2.5, 97.5])
        bootstrap_se = np.std(bootstrap_samples_mean_results, ddof=1)

        print(f"Exact mean from sample is {sum(observations)/len(observations):.3f}.\n"
              f"Through bootstrapping, we have estimated an standard deviation of {bootstrap_se:.3f}, "
              f"through our {num_bootstrap_samples} bootstrapped samples.\n"
              f"The 95% confidence interval falls between {lower:.3f} and {upper:.3f}.")

        plot_bootstraps = input("Do you want to plot a histogram of the bootstrapped samples? (y/n): ")
        if plot_bootstraps.lower() == "yes" or plot_bootstraps.lower() == "y":
            plt.hist(bootstrap_samples_mean_results, bins=20, color="orange")
            plt.axvline(bootstrap_mean, color="green", label="Mean")
            plt.axvline(lower, color="red", label="Bottom 2.5%")
            plt.axvline(upper, color="red", label="Upper 97.5%")
            plt.xlabel("Average for bootstrap sample")
            plt.ylabel("Num samples with corresponding mean")
            plt.title(f"Bootstrap for {label.lower()}")
            plt.show()

        else:
            print("Plotting declined by user.\n")

    def calculate_scipy_bootstrap_and_print(self, observations:list, label:str, num_bootstrap_samples:int=10000):
        # alternativ bootstrap med scipy (percentile metode)
        observations = np.array(observations)
        np.random.seed(777)
        bootstrap_result = stats.bootstrap((observations,), np.mean, n_resamples=num_bootstrap_samples, method="percentile")
        lower, upper = bootstrap_result.confidence_interval
        print(f"Exact mean from sample is {sum(observations)/len(observations):.3f}.\n"
              f"Scipy bootstrap ({num_bootstrap_samples} resamples, percentile method).\n"
              f"The 95% confidence interval falls between {lower:.3f} and {upper:.3f}.\n")

    def linear_regression(self):
        self.regression_calculations()
        lrm = smf.ols("floors_reached ~ ascension + damage_per_floor + necrobinder + silent + defect + regent + relics_act1 + run_time_per_floor", data=self.df).fit()
        print(lrm.summary())
        action = input("Want QQ plot of residuals (check normality)? (y/n): ")
        if action.lower() == "y" or action.lower() == "yes":
            sm.qqplot(lrm.resid, line="45")
            plt.title("QQ plot of OLS residuals")
            plt.show()
        action = input("Want prediction intervals for floors reached? (y/n): ")
        if action.lower() == "y" or action.lower() == "yes":
            self.prediction_intervals(lrm)
        action = input("Want personalized tips on how to play better? (y/n): ")
        if action.lower() == "y" or action.lower() == "yes":
            self.linear_regression_play_insights()

    def logistic_regression(self):
        self.regression_calculations()
        self.df["win"] = self.df["win"].astype(int)
        logistic_regression_model = smf.logit("win ~ ascension + damage_per_floor + relics_act1 + run_time_per_floor", data=self.df).fit()
        print(logistic_regression_model.summary())
        action = input("Want odds ratios? (y/n): ")
        if action.lower() == "y" or action.lower() == "yes":
            print(np.exp(logistic_regression_model.params)) # this is odd ratios : exp of log odds ratio returns odds ratio, the params are originally on log odds ratio scale

        action = input("Want to predict a specific run? (y/n): ")
        if action.lower() == "y" or action.lower() == "yes":
            try:
                ascension = int(input("Ascension: "))
                relics_act1 = int(input("Relics act 1: "))
                damage_per_floor = float(input("Damage per floor: "))
                run_time_per_floor = float(input("Run time per floor: "))
                prediction = logistic_regression_model.predict(pd.DataFrame({"ascension": [ascension],
                                                                             "relics_act1": [relics_act1],
                                                                             "damage_per_floor": [damage_per_floor],
                                                                             "run_time_per_floor": [run_time_per_floor]}))
                print(f"Predicted chance of winning given the parameters (and the data set) is {prediction.values[0]:.3f} ({prediction.values[0]*100:.1f}%)")
            except ValueError:
                print("Something went wrong, one of the values given were likely in wrong format.\n"
                      "Exiting logistic regression mode...")

        action = input("Want full model evaluation on a test set? (y/n): ")
        if action.lower() == "y" or action.lower() == "yes":
            self.evaluate_logistic_on_test()

    def prepare_classification_data(self):
        # lager train/test med samme variabler som logit (fra .run: ascension, damage, relics act1, run_time)
        self.regression_calculations()
        self.df["win"] = self.df["win"].astype(int)
        feature_cols = ["ascension", "damage_per_floor", "relics_act1", "run_time_per_floor"]
        model_df = self.df[feature_cols + ["win"]].copy()
        before = len(model_df)
        model_df = model_df.dropna()
        dropped = before - len(model_df)
        if dropped > 0:
            print(f"Dropped {dropped} runs with missing values (e.g. runs with no floors in map_point_history).\n")
        print(f"Train/test split: {len(model_df)} runs total (70% train, 30% test).\n")
        np.random.seed(777)
        n = len(model_df)
        train_mask = np.random.rand(n) < 0.7
        self._last_train_mask = train_mask
        train_df = model_df[train_mask]
        test_df = model_df[~train_mask]
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
        y_train = train_df["win"].values
        y_test = test_df["win"].values
        return train_df, test_df, X_train, X_test, y_train, y_test

    def print_classification_metrics(self, label, y_test, y_pred, y_prob):
        # modell evaluering: confusion matrix, sens/spec manuelt, accuracy, auc, brier
        print(f"\n--- {label} ---")
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion matrix:")
        print(cm)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        else:
            sensitivity = 0
            specificity = 0
            tp = fn = tn = fp = 0
        acc = accuracy_score(y_test, y_pred)
        print(f"Sensitivity (recall): {sensitivity:.3f}")
        print(f"Specificity: {specificity:.3f}")
        print(f"Accuracy: {acc:.3f}")
        auc = None
        brier = brier_score_loss(y_test, y_prob)
        print(f"Brier score: {brier:.3f}")
        fpr, tpr = None, None
        if len(np.unique(y_test)) < 2:
            print("Test set only has one class - skipping ROC and AUC.\n")
        else:
            auc = roc_auc_score(y_test, y_prob)
            print(f"AUC: {auc:.3f}")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            action = input("Show ROC plot for this model? (y/n): ")
            if action.lower() == "y" or action.lower() == "yes":
                plt.plot(fpr, tpr, label=label)
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("False positive rate")
                plt.ylabel("True positive rate")
                plt.title(f"ROC - {label}")
                plt.legend()
                plt.show()
        return {"accuracy": acc, "sensitivity": sensitivity, "specificity": specificity, "auc": auc, "brier": brier}, fpr, tpr

    def evaluate_logistic_on_test(self):
        # oppgave 6: logistisk regresjon evaluert på testsett
        train_df, test_df, X_train, X_test, y_train, y_test = self.prepare_classification_data()
        print(f"Training on {len(train_df)} runs, testing on {len(test_df)} runs.\n")
        logistic_model = smf.logit("win ~ ascension + damage_per_floor + relics_act1 + run_time_per_floor", data=train_df).fit()
        y_prob = logistic_model.predict(test_df)
        y_pred = (y_prob >= 0.5).astype(int)
        self.print_classification_metrics("Logistic regression", y_test, y_pred, y_prob)

    def naive_bayes_and_compare(self):
        # oppgave 7: gaussisk naive bayes + sammenligning med logistisk
        train_df, test_df, X_train, X_test, y_train, y_test = self.prepare_classification_data()
        print(f"Training on {len(train_df)} runs, testing on {len(test_df)} runs.\n")
        logistic_model = smf.logit("win ~ ascension + damage_per_floor + relics_act1 + run_time_per_floor", data=train_df).fit()
        log_prob = logistic_model.predict(test_df)
        log_pred = (log_prob >= 0.5).astype(int)
        log_metrics, log_fpr, log_tpr = self.print_classification_metrics("Logistic regression", y_test, log_pred, log_prob)
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        gnb_pred = gnb.predict(X_test)
        gnb_prob = gnb.predict_proba(X_test)[:, 1]
        gnb_metrics, gnb_fpr, gnb_tpr = self.print_classification_metrics("Naive Bayes", y_test, gnb_pred, gnb_prob)
        print("\nComparison (test set):")
        print(f"{'Model':<22} {'Accuracy':>10} {'Sensitivity':>13} {'Specificity':>13} {'AUC':>8} {'Brier':>8}")
        for name, m in [("Logistic reg.", log_metrics), ("Naive Bayes", gnb_metrics)]:
            auc_str = f"{m['auc']:.3f}" if m["auc"] is not None else "n/a"
            print(f"{name:<22} {m['accuracy']:>10.3f} {m['sensitivity']:>13.3f} {m['specificity']:>13.3f} {auc_str:>8} {m['brier']:>8.3f}")
        if log_fpr is not None and gnb_fpr is not None:
            action = input("Show both ROC curves in same plot? (y/n): ")
            if action.lower() == "y" or action.lower() == "yes":
                plt.plot(log_fpr, log_tpr, label="Logistic regression")
                plt.plot(gnb_fpr, gnb_tpr, label="Naive Bayes")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("False positive rate")
                plt.ylabel("True positive rate")
                plt.title("ROC - logistic vs Naive Bayes")
                plt.legend()
                plt.show()

    def permutation_test_damage(self):
        # oppgave 8: permutasjonstest
        # sammenligner med welch t-test
        print("1. 'total_damage' (winning vs losing runs)\n"
              "2. 'relics_act1' (relics by end of act 1)")
        choice = input("Which comparison? (enter for total_damage) ")
        if len(choice) == 0 or choice.lower() == "total_damage":
            group_a = np.array(self.damage_winning_runs)
            group_b = np.array(self.damage_losing_runs)
            label = "total damage (winning minus losing)"
        else:
            group_a = np.array(self.act1_relics_winning)
            group_b = np.array(self.act1_relics_losing)
            label = "act 1 relics (winning minus losing)"
        if len(group_a) == 0 or len(group_b) == 0:
            print("Not enough data in one of the groups.\n")
            return
        observed = np.mean(group_a) - np.mean(group_b)
        combined = np.concatenate([group_a, group_b])
        n_a = len(group_a)
        n_b = len(group_b)
        np.random.seed(777)
        perm_diffs = []
        num_permutations = 10000
        for _ in range(num_permutations):
            shuffled = np.random.permutation(combined)
            perm_a = shuffled[:n_a]
            perm_b = shuffled[n_a:]
            perm_diffs.append(np.mean(perm_a) - np.mean(perm_b))
        perm_diffs = np.array(perm_diffs)
        p_perm = np.mean(np.abs(perm_diffs) >= np.abs(observed))
        t_stat, p_ttest = stats.ttest_ind(group_a, group_b, equal_var=False)
        print(f"\n{label.capitalize()}")
        print(f"Observed difference (winning - losing): {observed:.3f}")
        print(f"Permutation p-value ({num_permutations} iterations, two-sided): {p_perm:.3f}")
        print(f"Welch t-test p-value: {p_ttest:.3f} (T={t_stat:.3f})\n")
        action = input("Plot histogram of permuted differences? (y/n): ")
        if action.lower() == "y" or action.lower() == "yes":
            plt.hist(perm_diffs, bins=30, color="steelblue")
            plt.axvline(observed, color="red", label="Observed difference")
            plt.xlabel("Permuted mean difference")
            plt.ylabel("Count")
            plt.title(f"Permutation test - {label}")
            plt.legend()
            plt.show()

    def prediction_intervals(self, lrm=None):
        # oppgave 9: prediksjonsintervaller med get_prediction().summary_frame()
        self.regression_calculations()
        if lrm is None:
            lrm = smf.ols("floors_reached ~ ascension + damage_per_floor + necrobinder + silent + defect + regent + relics_act1 + run_time_per_floor", data=self.df).fit()
        pred_frame = lrm.get_prediction().summary_frame()
        print("In-sample prediction frame (first rows):")
        print(pred_frame.head())
        # hypotetiske scenario: varierer ascension, resten på median fra datasettet
        med_damage = self.df["damage_per_floor"].median()
        med_relics = self.df["relics_act1"].median()
        med_time = self.df["run_time_per_floor"].median()
        max_asc = int(self.df["ascension"].max())
        hypo = pd.DataFrame({
            "ascension": list(range(0, max_asc + 1)),
            "damage_per_floor": med_damage,
            "necrobinder": 0,
            "silent": 0,
            "defect": 0,
            "regent": 0,
            "relics_act1": med_relics,
            "run_time_per_floor": med_time,
        })
        hypo_pred = lrm.get_prediction(hypo).summary_frame()
        print("\nHypothetical predictions (ascension 0–{}, other vars at median):".format(max_asc))
        print(hypo_pred)
        x = hypo["ascension"]
        plt.plot(x, hypo_pred["mean"], color="blue", label="Predicted floors")
        plt.fill_between(x, hypo_pred["mean_ci_lower"], hypo_pred["mean_ci_upper"], alpha=0.3, label="Mean CI")
        plt.fill_between(x, hypo_pred["obs_ci_lower"], hypo_pred["obs_ci_upper"], alpha=0.2, label="Prediction interval")
        plt.xlabel("Ascension")
        plt.ylabel("Floors reached")
        plt.title("Predicted floors vs ascension (other covariates at median)")
        plt.legend()
        action = input("Show plot? (y/n): ")
        if action.lower() == "y" or action.lower() == "yes":
            plt.show()
        else:
            print("Plotting declined by user.\n")

    def regression_calculations(self):
        floors_reached = []

        # CHARACTERS
        ironclad = []  # baseline, will not be added and ignored
        necrobinder = []
        silent = []
        defect = []
        regent = []

        damage_per_floor = []
        number_of_relics = []

        for run_number, value in self.runs.items():
            floors_reached.append(value[3])
            if value[0].lower() == "necrobinder":
                necrobinder.append(1)
                silent.append(0)
                defect.append(0)
                regent.append(0)
            elif value[0].lower() == "silent":
                necrobinder.append(0)
                silent.append(1)
                defect.append(0)
                regent.append(0)
            elif value[0].lower() == "defect":
                necrobinder.append(0)
                silent.append(0)
                defect.append(1)
                regent.append(0)
            elif value[0].lower() == "regent":
                necrobinder.append(0)
                silent.append(0)
                defect.append(0)
                regent.append(1)
            else:
                necrobinder.append(0)
                silent.append(0)
                defect.append(0)
                regent.append(0)
            # noen runs har 0 rom i map_point_history - da finnes ikke index 5
            if len(value) > 5:
                damage_per_floor.append(value[5])
            else:
                damage_per_floor.append(np.nan)


            number_of_relics.append(len(self.all_runs_relics[run_number]))

        self.df['floors_reached'] = floors_reached
        self.df['necrobinder'] = necrobinder
        self.df['silent'] = silent
        self.df['defect'] = defect
        self.df['regent'] = regent
        self.df['damage_per_floor'] = damage_per_floor
        self.df['relics_act1'] = self.all_runs_and_relics_in_act1
        self.df["damage_per_floor"] = pd.to_numeric(self.df["damage_per_floor"], errors="coerce")
        self.df['floors_reached'] = self.df["floors_reached"].replace(0,1) # some runs dont have any floors reached
        self.df['run_time_per_floor'] = self.df['run_time'] / self.df['floors_reached']

        self.df["gold_spent"] = self.run_gold_spent
        self.df["gold_gained"] = self.run_gold_gained
        self.df["hp_healed"] = self.run_hp_healed
        self.df["rest_sites"] = self.run_rest_sites
        self.df["shop_visits"] = self.run_shop_visits
        self.df["elite_fights"] = self.run_elite_fights
        self.df["boss_fights"] = self.run_boss_fights
        self.df["monster_rooms"] = self.run_monster_rooms
        self.df["deck_size"] = self.run_deck_size
        self.df["total_relics"] = [len(self.all_runs_relics[i]) for i in range(len(self.df))]
        floors_safe = self.df["floors_reached"].replace(0, 1)
        self.df["heal_per_floor"] = self.df["hp_healed"] / floors_safe
        self.df["gold_per_floor"] = self.df["gold_gained"] / floors_safe

    def print_ols_coaching_tips(self, model, var_labels, outcome_label="etasjer"):
        # oversetter koeffisienter til noe en kan bruke i neste run
        print(f"\n--- Hva som henger sammen med {outcome_label} (p < 0.05) ---\n")
        found_any = False

        for var in model.params.index:
            if var == "Intercept":
                continue
            pval = model.pvalues[var]
            if pval >= 0.05:
                continue
            found_any = True
            coef = model.params[var]
            label = var_labels.get(var, var.replace("_", " "))
            if coef > 0:
                print(f" - Jo mer {label}, desto høyere {outcome_label} i snitt (ca +{coef:.2f} per enhet, p={pval:.3f}).")
            else:
                print(f" - Høyere {label} henger sammen med lavere {outcome_label} (ca {coef:.2f} per enhet, p={pval:.3f}).")
        if not found_any:
            print("Ingen tydelige enkeltfaktorer på 5% nivå - vanskelig datasett eller for få runs.\n")

    def print_personal_profile_vs_winners(self):
        self.regression_calculations()
        wins = self.df[self.df["win"] == True]
        losses = self.df[self.df["win"] == False]
        if len(wins) == 0:
            print("Du har ingen registrerte seire ennå - tipsene blir mer presise etter hvert!\n")
            wins = self.df
        print("\n--- Din profil vs gjennomsnittet på vinnende runs ---\n")
        tips = []
        metrics = [
            ("rest_sites", "rest sites besøkt"),
            ("relics_act1", "relics ved slutten av act 1"),
            ("gold_spent", "gull brukt totalt"),
            ("hp_healed", "HP healet totalt"),
            ("elite_fights", "elite-kamper"),
            ("damage_per_floor", "skade per etasje"),
            ("deck_size", "kort i deck ved slutt"),
            ("shop_visits", "butikkbesøk"),
        ]

        for col, name in metrics:
            if col not in self.df.columns:
                continue
            yours = self.df[col].mean()
            winner_avg = wins[col].mean()
            diff = winner_avg - yours
            print(f"  {name.capitalize()}: gjennomsnittlig ca {yours:.1f}, men på vinnende ca {winner_avg:.1f} (forskjell {diff:+.1f})")
            if abs(diff) > 0.01 * max(abs(winner_avg), 1):
                if diff > 0:
                    tips.append(f"prøv å øke {name} mot nivået vinnende runs har (~{winner_avg:.1f})")
                else:
                    tips.append(f"du ligger over vinnere på {name} - kanskje du kan trimme litt?")
        if len(losses) > 0 and len(wins) > 0:
            print(f"\n  Du vinner ca {100*len(wins)/len(self.df):.1f}% av {len(self.df)} runs i denne mappen.")
        if tips:
            print("\Et par idéer til neste run:")
            for t in tips[:5]:
                print(f" – {t.capitalize()}.")

    def print_elite_death_guide(self):
        # hvilke elites dreper deg oftest (fra killed_by_encounter i .run)
        losses = self.df[self.df["win"] == False]
        elites = losses["killed_by_encounter"].value_counts()
        elites = elites[elites.index.astype(str).str.contains("ELITE", case=False, na=False)]
        if len(elites) == 0:
            print("\nIngen elite-død registrert i encounter-feltet (eller du vinner dem!).\n")
            return
        print("\n--- Elites som oftest ender runnet ditt ---\n")
        for enc, count in elites.head(6).items():
            name = enc.split(".")[1].replace("_", " ").lower().capitalize() if "." in str(enc) else str(enc)
            print(f" - {name}: {count} ganger")

    def linear_regression_play_insights(self):
        # lineær regresjon med flere .run-variabler + menneskelige tips
        self.regression_calculations()
        print("\n" + "="*60)
        print(" SPILL SMARTERE - statistikk fra dine .run-filer ")
        print("="*60)
        self.print_personal_profile_vs_winners()
        self.print_elite_death_guide()
        model_df = self.df.dropna(subset=["floors_reached", "damage_per_floor"])
        if len(model_df) < 10:
            print("For få runs med full historikk til å stole på regresjonen.\n")
            return
        # hovedmodell: hvor langt kommer du?
        formula_main = ("floors_reached ~ rest_sites + relics_act1 + gold_spent + hp_healed + "
                        "elite_fights + ascension + damage_per_floor + shop_visits + deck_size + "
                        "necrobinder + silent + defect + regent")

        lrm_main = smf.ols(formula_main, data=model_df).fit()
        print("\n--- Modell 1: Hva predikerer hvor mange etasjer du når? ---\n")
        print(lrm_main.summary())

        var_labels = {
            "rest_sites": "hvilesteder",
            "relics_act1": "relics i act 1",
            "gold_spent": "gull brukt",
            "hp_healed": "HP healet",
            "elite_fights": "elite-kamper",
            "ascension": "ascension-nivå",
            "damage_per_floor": "skade per etasje",
            "shop_visits": "butikker",
            "deck_size": "deck-størrelse",
            "necrobinder": "Necrobinder (vs Ironclad)",
            "silent": "Silent",
            "defect": "Defect",
            "regent": "Regent",
        }

        self.print_ols_coaching_tips(lrm_main, var_labels)
        r2 = lrm_main.rsquared
        print(f"\n  Modellen forklarer ca {r2*100:.1f}% av variasjonen i etasjer (R2).\n"
              f"  Merk: seire er nesten alltid like lange (du vinner = full run), så mye av forskjellen er tap vs seier.\n")
        # modell 2: bare tapte runs - der varierer etasjer (tidlig død vs act 3). Seire gir ikke mening her.
        losses_only = model_df[model_df["win"] == False]
        if len(losses_only) >= 10:
            formula_losses = ("floors_reached ~ rest_sites + relics_act1 + gold_spent + hp_healed + "
                              "elite_fights + ascension + damage_per_floor + deck_size")
            lrm_losses = smf.ols(formula_losses, data=losses_only).fit()
            print("\n--- Modell 2: Når du taper - hva er det som henger sammen med å overleve lenger? ---\n")
            print(lrm_losses.summary())
            self.print_ols_coaching_tips(lrm_losses, var_labels, outcome_label="etasjer før du dør")
        else:
            print("\n--- Modell 2: Tapte runs ---\n")
            print(f"  For få tapte runs ({len(losses_only)}) til regresjon her - spill litt mer!\n")
        # modell 3: skade per etasje
        formula_damage = "damage_per_floor ~ rest_sites + relics_act1 + elite_fights + ascension + hp_healed"
        lrm_dmg = smf.ols(formula_damage, data=model_df).fit()
        print("\n--- Modell 3: Hva avgjør skade per etasje? (lavere kan bety smartere pathing) ---\n")
        print(lrm_dmg.summary())
        dmg_labels = {k: v for k, v in var_labels.items() if k in lrm_dmg.params.index}
        self.print_ols_coaching_tips(lrm_dmg, dmg_labels, outcome_label="skade per etasje")
        if "rest_sites" in lrm_main.params and lrm_main.pvalues["rest_sites"] < 0.1:
            coef_rest = lrm_main.params["rest_sites"]
            med_rest = model_df["rest_sites"].median()
            print(f"\n  Tenkeøvelse: modellen sier én ekstra hvile utover median (~{med_rest:.0f}) "
                  f"kan bety ca {coef_rest:+.1f} etasjer — hvis du ofte skipper camp.\n")
        print("\n--- Karakterer: snitt etasjer og win rate (dine runs) ---\n")
        for char, vals in self.all_played_characters.items():
            if len(vals) < 3:
                continue
            sub = None
            c_lower = char.lower()
            if c_lower == "ironclad":
                sub = model_df[(model_df["necrobinder"]==0) & (model_df["silent"]==0) & (model_df["defect"]==0) & (model_df["regent"]==0)]
            elif c_lower == "necrobinder":
                sub = model_df[model_df["necrobinder"]==1]
            elif c_lower == "silent":
                sub = model_df[model_df["silent"]==1]
            elif c_lower == "defect":
                sub = model_df[model_df["defect"]==1]
            elif c_lower == "regent":
                sub = model_df[model_df["regent"]==1]
            if sub is not None and len(sub) > 0:
                avg_floors = sub["floors_reached"].mean()
                print(f"  {char.lower().capitalize()}: {vals[2]:.1%} win rate, ~{avg_floors:.1f} etasjer i snitt ({vals[0]} runs)")

        action = input("\nPlot etasjer vs hvilesteder (ett punkt per run)? (y/n): ")

        if action.lower() == "y" or action.lower() == "yes":
            colors = ["green" if w else "crimson" for w in model_df["win"]]
            plt.scatter(model_df["rest_sites"], model_df["floors_reached"], c=colors, alpha=0.6)
            plt.xlabel("Hvilesteder besøkt")
            plt.ylabel("Etasjer nådd")
            plt.title("Dine runs: hvile vs. dybde (grønn=seier, rød=tap)")
            plt.show()
        action = input("Plot relics act 1 vs. etasjer? (y/n): ")
        if action.lower() == "y" or action.lower() == "yes":
            plt.scatter(model_df["relics_act1"], model_df["floors_reached"], c=colors, alpha=0.6)
            plt.xlabel("Relics ved slutten av act 1")
            plt.ylabel("Etasjer nådd")
            plt.title("Relics tidlig vs. hvor langt du kom")
            plt.show()
        print("\n  Lykke til på neste run - tallene er fra DINE filer, ikke en guide for alle.\n")

    def calculate_initial_statistics(self):
        self.overall_win_rate = self.df["win"].mean()
        self.total_wins = self.df["win"].sum()

        for run_number, character in enumerate(self.df["players"]):
            #print(character)
            run = []
            # adding name as first element in the run list
            character_name = character[0]['character']
            if character_name.startswith("CHARACTER."):
                character_name = character_name[10:]
                #print(character_name)
            run.append(character_name) # CHARACTER NAME: INDEX 0
            if character_name not in self.all_played_characters.keys():
                self.all_played_characters[character_name] = [0]
            self.all_played_characters[character_name][0] += 1
            self.runs[run_number] = run
        for run_number, win_status in enumerate(self.df["win"]):
            self.runs[run_number].append(win_status) # WIN BOOLEAN: INDEX 1
            character_name = self.df.iloc[run_number]["players"][0]["character"][10:]
            if len(self.all_played_characters[character_name]) == 1:
                self.all_played_characters[character_name].append(0)
            if win_status:
                self.all_played_characters[character_name][1] += 1
            self.total_runs += 1
        run_number = 0
        for ascension in self.df["ascension"]:
            self.runs[run_number].append(ascension) # ASCENSION NUMBER: INDEX 2
            run_number += 1
        for run_number, map_point_history in enumerate(self.df["map_point_history"]):
            relics = self.df.iloc[run_number]['players'][0]['relics']
            self.all_runs_relics[run_number] = relics

            # map_point_history er liste over acts, hvert act er liste over rom med player_stats
            all_rooms = [room for act in map_point_history for room in act]
            num_rooms = len(all_rooms)
            self.runs[run_number].append(num_rooms)  # NUMBER OF ROOMS: INDEX 3

            all_damage = [room["player_stats"][0]["damage_taken"] for act in map_point_history for room in act]
            total_damage = sum(all_damage)
            self.runs[run_number].append(total_damage) # TOTAL DAMAGE TAKEN: INDEX 4

            # summerer player_stats over alle rom (gull, heal, skade)
            ps_list = [room["player_stats"][0] for act in map_point_history for room in act]
            self.run_gold_spent.append(sum(p.get("gold_spent", 0) for p in ps_list))
            self.run_gold_gained.append(sum(p.get("gold_gained", 0) for p in ps_list))
            self.run_hp_healed.append(sum(p.get("hp_healed", 0) for p in ps_list))
            rest_sites = 0
            shop_visits = 0
            elite_fights = 0
            boss_fights = 0
            monster_rooms = 0
            for room in all_rooms:
                mpt = str(room.get("map_point_type", "")).lower()
                if "rest" in mpt:
                    rest_sites += 1
                elif "shop" in mpt:
                    shop_visits += 1
                elif "elite" in mpt:
                    elite_fights += 1
                elif "boss" in mpt:
                    boss_fights += 1
                elif "monster" in mpt:
                    monster_rooms += 1
            self.run_rest_sites.append(rest_sites)
            self.run_shop_visits.append(shop_visits)
            self.run_elite_fights.append(elite_fights)
            self.run_boss_fights.append(boss_fights)
            self.run_monster_rooms.append(monster_rooms)
            deck = self.df.iloc[run_number]["players"][0].get("deck", [])
            self.run_deck_size.append(len(deck) if deck is not None else 0)

            self.total_damage_taken += total_damage

            self.all_runs_and_relics_in_act1.append(len([item for item in relics if item['floor_added_to_deck'] <= 16]))
            if self.runs[run_number][1]:
                self.act1_relics_winning.append(len([item for item in relics if item['floor_added_to_deck'] <= 16]))
            else:
                if self.runs[run_number][3] >= 16:  # if we add everything, this data will be of much less value -
                    # as runs losing before act 1 naturally will have fewer relics
                    self.act1_relics_losing.append(len([item for item in relics if item['floor_added_to_deck'] <= 16]))

            if num_rooms != 0:
                self.runs[run_number].append(total_damage/num_rooms) # AVERAGE DAMAGE TAKEN: INDEX 5

            if self.runs[run_number][1]:
                self.damage_winning_runs.append(total_damage)
                self.damage_per_floor_winning_runs.append(all_damage)
            else:
                self.damage_losing_runs.append(total_damage)
                self.damage_per_floor_losing_runs.append(all_damage)

        self.damage_per_floor_winning_runs = [damage for all_damage in self.damage_per_floor_winning_runs for damage in all_damage]
        self.damage_per_floor_losing_runs = [damage for all_damage in self.damage_per_floor_losing_runs for damage in all_damage]

        for run_number, killed_by_encounter in enumerate(self.df["killed_by_encounter"]):
            self.runs[run_number].append(killed_by_encounter) # KILLED BY ENCOUNTER: INDEX 6

        for run_number, killed_by_event in enumerate(self.df["killed_by_event"]):
            self.runs[run_number].append(killed_by_event) # KILLED BY EVENT: INDEX 7

        #for key, value in self.runs.items():
        #    print(key, ": ", value)

        for character, values in self.all_played_characters.items():
            if values[0] != 0:
                win_rate = int(values[1])/int(values[0])
                values.append(win_rate)

        if self.total_runs != 0:
            self.average_damage_taken = self.total_damage_taken/self.total_runs

        self.df = self.df.drop(['build_id', 'game_mode', 'start_time',
                                'was_abandoned', 'seed', 'schema_version',
                                'modifiers', 'platform_type'], axis=1)
        pd.set_option('display.max_columns', None)
        print(self.df)












if __name__ == "__main__":
    si = StatsInterface("runs")
    si.load_dataframe()
    si.calculate_initial_statistics()
    si.start_inferface()