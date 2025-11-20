import os

import numpy as np

from matplotlib import pyplot as plt
import math

source_path = "/home/ensu/Documents/weird/IsaacLab/logs/result_trash"


def viz_object_result(object_success, object_dev, image_name="result"):
    os.makedirs(image_name, exist_ok=True)

    all_types_object_success = {}
    all_success_dict = {}

    for type_name in object_success.keys():
        print(f"Visualizing {type_name} results")
        type_data = object_success[type_name]
        all_types_object_success[type_name] = {}
        for per_result in type_data:
            for object_name, values in per_result.items():
                if object_name not in all_types_object_success[type_name]:
                    all_types_object_success[type_name][object_name] = []
                all_types_object_success[type_name][object_name].append(values)

        object_names = list(all_types_object_success[type_name].keys())
        if len(object_names) == 0:
            continue

        num_objects = len(object_names)
        ncols = math.ceil(math.sqrt(num_objects))
        nrows = math.ceil(num_objects / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        axes = axes.flatten()

        all_success = None

        for i in range(num_objects):
            ax = axes[i]

            results = all_types_object_success[type_name][object_names[i]]
            all_result_len = [len(r) for r in results]
            all_result = np.zeros((len(all_result_len), max(all_result_len)))
            all_result[
                :,
            ] = all_types_object_success[type_name][object_names[i]][np.argmax(
                all_result_len)]
            all_result[
                :,
            ] = all_types_object_success[type_name][object_names[i]][np.argmax(
                all_result_len)]
            for j in range(len(results)):
                obj_result = results[j]
                all_result[j, :len(obj_result)] = obj_result
            if all_success is None:
                all_success = np.zeros(
                    (len(all_result_len), max(all_result_len)))

            all_success += all_result

            # for train_id, training_result in enumerate(
            #         all_types_object_success[type_name][object_names[i]]):

            #     ax.plot(training_result, label=f"train_{train_id}")
            #     ax.set_title(object_names[i])
            #     ax.set_xlim(0, len(training_result) + 1)

            ax.plot(np.min(all_result, 0), color='orange', label="mean")
            ax.set_title(object_names[i])
            ax.set_xlim(0, all_result.shape[-1] + 1)
            ax.legend()
            ax.grid(True)
        plt.suptitle(f"{type_name} object success", fontsize=36)
        plt.savefig(f"{image_name}/{type_name}_object_success.png")

        # plt.show()
        plt.close()
        plt.cla()
        plt.clf()

        all_success_dict[type_name] = all_success.min(0) / num_objects

    for type_name, all_success in all_success_dict.items():
        plt.plot(all_success[:40], label=type_name)

        plt.grid(True)
        plt.suptitle(f"{type_name} all Success Rate", fontsize=16)
        plt.savefig(f"{image_name}/all_{type_name}_success.png",
                    bbox_inches='tight')
        plt.close()
        plt.cla()
        plt.clf()

    for type_name, all_success in all_success_dict.items():

        plt.plot(all_success[:40], label=type_name)

        plt.grid(True)
        print(type_name, all_success[min(40, len(all_success) - 1)])
    plt.suptitle("All Success Rate", fontsize=16)

    # Move legend outside to the right
    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12)

    # Adjust the layout manually to avoid squeezing
    plt.subplots_adjust(right=0.75)

    # Save with tight bounding box
    plt.savefig(f"{image_name}/all_success.png", bbox_inches='tight')
    plt.close()


object_success = {}
object_reward = {}
whole_success = {}
whole_reward = {}
object_dev = {}
result_type = os.listdir(
    "/home/ensu/Documents/weird/IsaacLab/logs/result_trash")

for result in result_type:
    all_resutls = os.listdir(
        f"/home/ensu/Documents/weird/IsaacLab/logs/result_trash/{result}")
    object_success[result] = []
    object_reward[result] = []
    whole_success[result] = []
    whole_reward[result] = []
    object_dev[result] = []

    for subresult in all_resutls:
        per_result_folder = os.path.join(
            f"/home/ensu/Documents/weird/IsaacLab/logs/result_trash/{result}",
            subresult + "/eval_images",
        )
        if not os.path.exists(per_result_folder):
            continue
        per_result = os.listdir(per_result_folder)

        for file in per_result:
            if file.endswith(".npz"):

                if "object_success" in file:
                    data = np.load(os.path.join(per_result_folder, file))
                    data_dict = {key: data[key] for key in data.files}
                    object_success[result].append(data_dict)
                if "object_rollout" in file:
                    data = np.load(os.path.join(per_result_folder, file))
                    data_dict = {key: data[key] for key in data.files}
                    object_reward[result].append(data_dict)
                if "hand_success" in file:
                    data = np.load(os.path.join(per_result_folder, file))
                    whole_success[result].append(data["arr_0"])
                if "hand_rollout" in file:
                    data = np.load(os.path.join(per_result_folder, file))
                    whole_reward[result].append(data["arr_0"])
                if "hand_dev" in file:
                    data = np.load(os.path.join(per_result_folder, file))
                    data_dict = {key: data[key] for key in data.files}
                    object_dev[result].append(data_dict)

viz_object_result(object_success,
                  object_dev,
                  image_name=f"{source_path}/all_result/object_success")
