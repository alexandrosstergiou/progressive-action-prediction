'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import coloredlogs, logging
coloredlogs.install()

'''
---  S T A R T  O F  F U N C T I O N  G E T _ C O N F I G ---
    [About]
        Function for configuring the number of classes based on the dataset used.
    [Args]
        - name: String for the dataset name to be used for training.
    [Returns]
        - config: Dictionary that includes a `num_classes` term that specifies how many classes the dataset has.
'''
def get_config(name):

    config = {}
    # Case 1: UCF-101
    if 'UCF-101' in name.upper():
        config['num_classes'] = 101
        config['include_timeslices'] = False
    # Case 2: HMDB-51
    elif 'HMDB51' in name.upper():
        config['num_classes'] = 51
        config['include_timeslices'] = False
    # Case 3: Kinetics - if no variant is used, the
    elif 'KINETICS' in name.upper():
        if 'MINI' in name.upper():
            config['num_classes'] = 200
            config['include_timeslices'] = True
        elif '400' in name.upper():
            config['num_classes'] = 400
            config['include_timeslices'] = True
        elif '600' in name.upper():
            config['num_classes'] = 600
            config['include_timeslices'] = True
        elif '700' in name.upper():
            config['num_classes'] = 700
            config['include_timeslices'] = True
        else:
            logging.warning('Kinetics varinat not specified and/or supported initializing with 400 classes')
            config['num_classes'] = 400
            config['include_timeslices'] = True
    # Case 4: Diving 48
    elif 'CHARADES' in name.upper():
        config['num_classes'] = 157
        config['include_timeslices'] = False
    # Case 5: Moments in Time
    elif 'MOMENTS' in name.upper():
        config['num_classes'] = 339
        config['include_timeslices'] = True
    # Case 6: HACS
    elif 'HACS' in name.upper():
        config['num_classes'] = 200
        config['include_timeslices'] = True
    # Case 7: Something SOMETHING
    elif 'SMTH' in name.upper():
        if 'COARSE' in name.upper():
            config['num_classes'] = 50
            config['include_timeslices'] = False
        else:
            config['num_classes'] = 174
            config['include_timeslices'] = False
    else:
        logging.error("Configs for dataset '{}'' not found/supported. See supported datasets in `train.py`".format(name))
        raise NotImplemented

    logging.debug("Target dataset: '{}', configs: {}".format(name.upper(), config))

    return config
'''
---  E N D  O F  F U N C T I O N  G E T _ C O N F I G ---
'''
