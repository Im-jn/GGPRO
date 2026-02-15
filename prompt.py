import dgl
from config import CONFIG


def poi_augment(poi_base, user_id, graph, graph_nodes, imp_nodes, traj, time_stamps, next_time, last_traj_ptr=None,
                with_answer_instruction=True, with_hint=True):
    # default with the complete mode
    info, hint, history, question, answer = "", "", "", "", ""
    poi_set = set()  # to record all the poi used in this prompt
    # 1. process hint
    hint += "<Hint> Here are some detailed POIs information: \n "
    traj_info = poi_base.get_poi_info(traj).assign(new_column_name="time_stamp")
    traj_info["time_stamp"] = time_stamps
    graph_prompt_list, graph_poi_set = gen_graph_prompt(graph, poi_base.get_poi_info(graph_nodes),
                                                        traj_info, imp_nodes)
    for i, graph_prompt in enumerate(graph_prompt_list):
        hint += "\t({}){};\n".format(i + 1, graph_prompt)
    poi_set = poi_set.union(graph_poi_set)

    # 2. process history
    history += "<History> "
    history += "The following data is a trajectory of user {}: \n".format(user_id)
    cur_list = poi_base.get_poi_info(traj[-CONFIG.history_visits:])
    cur_prompt, cur_poi_set = gen_traj_prompt(user_id, cur_list, time_stamps[-CONFIG.history_visits:])
    history += cur_prompt
    poi_set = poi_set.union(cur_poi_set)

    # # 2'. process history
    # history += "<History> "
    # history += "The following data is the recent check-ins of user {}: \n".format(user_id)
    # cur_list = poi_base.get_poi_info(traj[-CONFIG.recent_visits:])
    # cur_prompt, cur_poi_set = gen_traj_prompt(user_id, cur_list, time_stamps[-CONFIG.recent_visits:])
    # history += cur_prompt
    # poi_set = poi_set.union(cur_poi_set)

    # history_prompt = ""
    # if CONFIG.history_visits > CONFIG.recent_visits and with_hint and len(graph_poi_set) > 0:
    #     history += "\nAnd the historical trajectory before the recent check-ins: \n".format(user_id)
    #     cur_list = poi_base.get_poi_info(traj[-CONFIG.history_visits:-CONFIG.recent_visits])
    #     history_prompt, history_poi_set = gen_traj_prompt(user_id, cur_list, time_stamps[-CONFIG.history_visits:-CONFIG.recent_visits])
    #     history += history_prompt
    #     poi_set = poi_set.union(history_poi_set)

    # 3. process question
    question += "<Question> Given the data, at {}, " \
                "Which POI id will the user visit? Select exactly one POI id from the candidate list.".format(next_time.strftime('%Y-%m-%d %H:%M'))

    # 4. process answer instructions
    answer += "<Answer>: At {}, the user will visit POI id ".format(
        next_time.strftime('%Y-%m-%d %H:%M'))

    # 5. process info
    info = describe_poi_set(poi_base.get_poi_info(list(poi_set)))  # add POI introduction info in front

    # 6. combination
    if with_hint:
        prompt = info + hint + history + question
    else:
        prompt = info + history + question
    if with_answer_instruction:  # whether to include answer instruction
        prompt += answer

    description_pieces = {"graph": graph_prompt_list,
                          "history": cur_prompt}  # record all the necessary pieces
    return prompt, poi_set, description_pieces


def gen_traj_prompt(user_id, poi_list, time_stamps):
    # initializing poi description sequence
    poi_seq = [auto_poi(info, mode="simple") for _, info in poi_list.iterrows()]
    poi_set = set([info["id"] for _, info in poi_list.iterrows()])
    # forming sentence
    prompt = ""
    for ts, poi in zip(time_stamps, poi_seq):
        prompt += "At {}, user visited {}.".format(ts.strftime('%Y-%m-%d %H:%M'), poi)
    return prompt+"\n", poi_set


def describe_poi_set(poi_list):
    # prompt = "<Info> Here is a list of POIs(Point-of-Interests) that will be used in following content:\n"
    prompt = "<Candidates> Here is a list of candidate POIs(Point-of-Interests):\n"
    for _, info in poi_list.iterrows():
        prompt += auto_poi(info, mode="is") + "; "
    return prompt + "\n"


class appeared_poi:
    def __init__(self, nodes_info):
        self.pois = []
        self.nodes_info = nodes_info

    def append(self, poi_id):
        pid = self.nodes_info.iloc[poi_id]["id"]
        if pid not in self.pois:
            self.pois.append(pid)

    def cat(self, poi_list):
        id_list = [self.nodes_info.iloc[i]["id"] for i in poi_list]
        for pid in id_list:
            if pid not in self.pois:
                self.pois.append(pid)


def has_edge(graph, u, v, etype='same'):
    try:
        eids = graph.edge_ids(u, v, etype=etype)  # Check u -> v
        return True
    except dgl.DGLError:
        return False


def gen_graph_prompt(graph, nodes_info, traj_info, imp_nodes):
    '''
    If it is a POI visited by the user:
        1. What kind of POI is this;
        2. How many times the user visited it, and when was the last time;
        2'. Which POIs did you predict the user would visit after this one, and actually
        3. Which POIs did the user visit after it;
        4. Which POI did other users visit most after visiting it;
    If it is an extra POI:
        1. What kind of POI is this;
        2. It has the following relationship with the POIs visited by the user:
            a. Its distance to these POIs is within 100m
            b. Its category is the same as these POIs
            c. Most users chose to visit it after visiting a certain POI
            d. You predicted the user would arrive at it after visiting a certain POI, but actually the user did not
    '''
    graph_prompt_list = []
    ap_poi = appeared_poi(nodes_info)
    for node in imp_nodes:
        info = nodes_info.iloc[node]
        u_trans_out = graph.out_edges(node, form='uv', etype='u_trans')[1].tolist()
        u_trans_in = graph.in_edges(node, form='uv', etype='u_trans')[0].tolist()
        prompt = ""
        # 1. What kind of POI is this
        prompt += auto_poi(info, mode="intro")
        ap_poi.append(node)

        if len(u_trans_out) > 0 or len(u_trans_in) > 0:  # If it is a POI visited by the user
            # 2. How many times the user visited it, and when was the last time
            visits = traj_info[traj_info["id"] == info["id"]].shape[0]
            last_visit_time = traj_info[traj_info["id"] == info["id"]]["time_stamp"].max()
            try:
                prompt += ", user visited {} times, " \
                      "last on {}. ".format(visits, last_visit_time.strftime('%Y-%m-%d %H:%M'))
            except:
                prompt += ", user visited {} times. ".format(visits)
                # print("no time stamp found") 

            # 2'. Which POIs did you predict the user would visit after this one, and actually
            rec_poi = graph.out_edges(node, form='uv', etype='recommend')[1].tolist()
            if len(rec_poi) > 0:
                prompt += "You predicted user would visit after it ".format(fill_in_enumerate(nodes_info, rec_poi))
                correct_poi = list(set(rec_poi) & set(u_trans_out))
                if len(correct_poi) == len(u_trans_out):
                    prompt += ", and it's correct. "
                elif len(correct_poi) > 0:
                    prompt += ", and some are correct. "
                else:
                    prompt += ", but user did not. "

            # 3. Which POIs did the user visit after it
            if len(u_trans_out) > 0:
                prompt += "User visited {} after it. ".format(fill_in_enumerate(nodes_info, u_trans_out))
                ap_poi.cat(u_trans_out)
            else:
                prompt += "This is user's first visit. "

            # 4. Which POI did other users visit most after visiting it
            other_trans_poi = graph.out_edges(node, form='uv', etype='o_trans')[1].tolist()
            if len(other_trans_poi) > 0:
                prompt += "And most user visited "
                prompt += fill_in_enumerate(nodes_info, other_trans_poi)
                prompt += " after visiting it. "
                ap_poi.cat(other_trans_poi)
        else:  # If it is an extra POI
            # It has the following relationship with the POIs visited by the user
            prompt += ". "
            # a. Its distance to these POIs is within 100m
            near_poi = graph.out_edges(node, form='uv', etype='near')[1].tolist()
            if len(near_poi) > 0:
                prompt += "Near {}. ".format(fill_in_enumerate(nodes_info, near_poi))
                ap_poi.cat(near_poi)
            # b. Its category is the same as these POIs
            same_cate_poi = graph.out_edges(node, form='uv', etype='same')[1].tolist()
            if len(same_cate_poi) > 0:
                prompt += "Same category as {}. ".format(fill_in_enumerate(nodes_info, same_cate_poi))
                ap_poi.cat(same_cate_poi)
            # c. Most users chose to visit it after visiting a certain POI
            o_trans_in = graph.in_edges(node, form='uv', etype='o_trans')[0].tolist()
            if len(o_trans_in) > 0:
                prompt += "Some users visited it after {}. ".format(fill_in_enumerate(nodes_info, o_trans_in))
                ap_poi.cat(o_trans_in)
            # d. You predicted the user would arrive at it after visiting a certain POI, but actually the user did not
            recommend_poi = graph.out_edges(node, form='uv', etype='recommend')[1].tolist()
            if len(recommend_poi) > 0:
                prompt += "You predicted user would visit it after "
                prompt += fill_in_enumerate(nodes_info, recommend_poi)
                prompt += ", but user did not."
                ap_poi.cat(recommend_poi)
        graph_prompt_list.append(prompt)
    return graph_prompt_list, set(ap_poi.pois)


def fill_in_enumerate(nodes_info, poi_list):
    prompt = "{}".format(auto_poi(nodes_info.iloc[poi_list[0]], mode="simple"))
    for poi in poi_list[1:]:
        prompt += ", {}".format(auto_poi(nodes_info.iloc[poi], mode="simple"))
    return prompt


def name_id(info, quot=False):
    # auto describe a POI
    # id, poi_name, category, interest
    if quot:
        if info["poi_name"] != "Not Specified" and info["interest"] != "Not Specified":
            return "\'{}_{} ({} {})\'".format(
                underline(info["poi_name"]), info["id"], info["category"], info["interest"])
        elif info["poi_name"] != "Not Specified" and info["interest"] == "Not Specified":
            return "\'{}_{} ({})\'".format(
                underline(info["poi_name"]), info["id"], info["category"])
        else:
            return "\'POI_{} ({})\'".format(info["id"], info["category"])
    else:
        if info["poi_name"] != "Not Specified" and info["interest"] != "Not Specified":
            return "\'{}_{}\' (which is a {} {})".format(
                underline(info["poi_name"]), info["id"], info["category"], info["interest"])
        elif info["poi_name"] != "Not Specified" and info["interest"] == "Not Specified":
            return "\'{}_{}\' (which is a {})".format(
                underline(info["poi_name"]), info["id"], info["category"])
        else:
            return "\'POI_{}\' (which is a {})".format(
                info["id"], info["category"])


def auto_poi(info, mode=["simple", "complex", "intro", "is", ""][-1]):
    if mode == "simple":
        return "id {}".format(info["id"], info["category"])
    elif mode == "complex":
        return "POI id {} which is a {} and has category id {}".format(
            info["id"], info["category"], info["category_id"])
    elif mode == "intro":
        # return "POI id {}: Is a {} and has category id {}".format(
        #     info["id"], info["category"], info["category_id"])
        return "POI id {}: Is a {}".format(info["id"], info["category"])
    elif mode == "is":
        # return "POI id {} is a {} with category id {}".format(
        #     info["id"], info["category"], info["category_id"])
        return "id {} is a {}".format(info["id"], info["category"])
    else:
        raise "No such mode for auto_poi"


def intro_poi(info):
    return "POI id {}: Is a {} and has Category id {}".format(info["id"], info["category"], info["category_id"])


def underline(name):
    return name.replace(" ", "_")
