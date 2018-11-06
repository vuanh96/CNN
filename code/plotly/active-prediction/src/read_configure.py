import constant


def get_configure(config_file_path='config/config.txt'):
    config = {}
    try:
        with open(config_file_path, 'r') as file:
            for line in file:
                if line.strip().startswith(constant.COMMENT_CHAR):
                    pass
                else:
                    key = line.split(constant.ASSIGN_CHAR)[0]
                    value = line.split(constant.ASSIGN_CHAR)[1].strip().replace('\'', '').replace('"', "")
                    config[key] = value
        return config

    except IOError:
        return {}
