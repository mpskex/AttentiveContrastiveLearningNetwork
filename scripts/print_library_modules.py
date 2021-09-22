from fgvc_cl.utils.Register import REGISTRY

if __name__ == '__main__':
    print("======= LOADED MODULES =======")
    for _type in REGISTRY.keys():
        print("Type: %16s" % (_type))
        for name in REGISTRY[_type].keys():
            print("\tName: %16s" % (name))