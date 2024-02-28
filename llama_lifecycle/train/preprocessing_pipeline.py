import numpy as np
import pandas as pd

from datasets import Dataset as Dataset
from datasets import load_dataset as load_dataset

train_dataset = load_dataset("OpenAssistant/oasst1", split="train[:1%]")
df = train_dataset.to_pandas()

message_tree_ids = np.unique(np.array(df["message_tree_id"]))
messages: dict[str, list] = {"message_tree_id": [], "message_tree_text": []}

for message_tree_id in message_tree_ids:
    try:
        # look at all data for this message tree
        one_message_tree = df.query(
            f"message_tree_id == '{message_tree_id}'"
        ).sort_values("created_date")
        text = ""

        # root message
        text += "<human>: " + one_message_tree.iloc[0].text
        # find root message's children
        children = one_message_tree[
            one_message_tree.parent_id == one_message_tree.iloc[0].message_id
        ]
        # find root message's top ranked child:
        child = children[np.isclose(children["rank"], 0.0, rtol=1e-09, atol=1e-09)]
        text += "\n" + "<bot>: " + child.iloc[0].text

        # proceed through rest of the above message tree until completion
        flag = True
        while flag:
            try:
                # find next prompt
                children = one_message_tree[
                    one_message_tree.parent_id == child.message_id.iloc[0]
                ]
                text += (
                    "\n"
                    + "### USER: "
                    + one_message_tree.loc[children.index].iloc[0].text
                )

                # find next children
                children = one_message_tree[
                    one_message_tree.parent_id
                    == one_message_tree.loc[children.index].iloc[0].message_id
                ]
                # find top ranked child:
                child = children[
                    np.isclose(children["rank"], 0.0, rtol=1e-09, atol=1e-09)
                ]
                text += "\n" + "### ASSISTANT: " + child.iloc[0].text
            except ValueError:
                flag = False

        messages["message_tree_id"].append(message_tree_id)
        messages["message_tree_text"].append(text)

    except IndexError:
        pass

message_df = pd.DataFrame.from_dict(messages)
Dataset.from_pandas(message_df).save_to_disk("../../datasets/oasst1")
