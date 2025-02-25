"""
- String feature
- Permission feature
- Component feature
- Environmental feature
- Method opcode feature
- Method API feature
- Shared library function opcode feature
"""

import os
import time

import magic
import zipfile
import hashlib
from collections import defaultdict

from androguard.misc import AnalyzeAPK, APK
import lxml.etree as etree
from xml.dom import minidom

from elftools.elf.elffile import ELFFile
from elftools.common.py3compat import BytesIO
from capstone import Cs, CS_ARCH_ARM64, CS_ARCH_ARM, CS_MODE_ARM

from ...tools import utils
from sys import platform as _platform
from ...config import logging

if _platform == "linux" or _platform == "linux2":
    TMP_DIR = '/tmp/'
elif _platform == "win32" or _platform == "win64":
    TMP_DIR = 'C:\\TEMP\\'
logger = logging.getLogger('feature.multimodality')
current_dir = os.path.dirname(os.path.realpath(__file__))
API_LIST = [api.decode('utf-8') for api in \
            utils.read_txt(os.path.join(current_dir, 'res/api_list.txt'), mode='rb')]


def get_multimod_feature(apk_path, api_list, save_path):
    """
    get feature
    :param apk_path: an absolute apk path
    :param api_list: api list, api level 22 is considered
    :param save_path: a path for saving feature
    :return: (status, result_file_path), e.g., (True, back_path_name+apk_name+'.data')
    """
    try:
        print("Processing " + apk_path)
        start_time = time.time()
        data_list = []
        # permission, components, environment information
        data_list.append(get_dict_feature_xml(apk_path))
        # string, method api, method opcodes
        string_feature_dict, api_feature_dict, opcode_feature_dict = get_feature_dex(apk_path, api_list)
        data_list.extend([string_feature_dict, api_feature_dict, opcode_feature_dict])
        # shared library feature
        data_list.append(get_feature_shared_lib(apk_path))

        utils.dump_json(data_list, save_path)
    except Exception as e:
        return e
    else:
        return save_path


def get_dict_feature_xml(apk_path):
    """
    get requested feature from manifest file
    :param apk_path: absolute path of an apk file
    :return: a dict of elements {feature:occurrence, ..., }
    """
    permission_component_envinfo = defaultdict(int)
    xml_tmp_dir = os.path.join(TMP_DIR, 'xml_dir')
    if not os.path.exists(xml_tmp_dir):
        os.mkdir(xml_tmp_dir)
    apk_name = os.path.splitext(os.path.basename(apk_path))[0]
    try:
        apk_path = os.path.abspath(apk_path)
        a = APK(apk_path)
        f = open(os.path.join(xml_tmp_dir, apk_name + '.xml'), 'wb')
        xmlstreaming = etree.tostring(a.xml['AndroidManifest.xml'], pretty_print=True, encoding='utf-8')
        f.write(xmlstreaming)
        f.close()
    except Exception as e:
        raise Exception("Fail to load xml file of apk {}:{}".format(apk_path, str(e)))

    # start obtain feature permission, components, environment information
    try:
        with open(os.path.join(xml_tmp_dir, apk_name + '.xml'), 'rb') as f:
            dom_xml = minidom.parse(f)
            dom_elements = dom_xml.documentElement

            dom_permissions = dom_elements.getElementsByTagName('uses-permission')
            for permission in dom_permissions:
                if permission.hasAttribute('android:name'):
                    permission_component_envinfo[permission.getAttribute('android:name')] = 1

            dom_activities = dom_elements.getElementsByTagName('activity')
            for activity in dom_activities:
                if activity.hasAttribute('android:name'):
                    permission_component_envinfo[activity.getAttribute('android:name')] = 1

            dom_services = dom_elements.getElementsByTagName("service")
            for service in dom_services:
                if service.hasAttribute("android:name"):
                    permission_component_envinfo[service.getAttribute("android:name")] = 1

            dom_contentproviders = dom_elements.getElementsByTagName("provider")
            for provider in dom_contentproviders:
                if provider.hasAttribute("android:name"):
                    permission_component_envinfo[provider.getAttribute("android:name")] = 1
                # uri
                dom_uris = provider.getElementsByTagName('grant-uri-permission')
                # intents --- action
                intent_actions = provider.getElementsByTagName('action')
                # we neglect to compose the uri feature by so-called paired provider name and intents
                # instead, the path of uri is used directly
                for uri in dom_uris:
                    if uri.hasAttribute('android:path'):
                        permission_component_envinfo[uri.getAttribute('android:path')] = 1

            dom_broadcastreceivers = dom_elements.getElementsByTagName("receiver")
            for receiver in dom_broadcastreceivers:
                if receiver.hasAttribute("android:name"):
                    permission_component_envinfo[receiver.getAttribute("android:name")] = 1

            dom_intentfilter_actions = dom_elements.getElementsByTagName("action")
            for action in dom_intentfilter_actions:
                if action.hasAttribute("android:name"):
                    permission_component_envinfo[action.getAttribute("android:name")] = 1

            dom_hardwares = dom_elements.getElementsByTagName("uses-feature")
            for hardware in dom_hardwares:
                if hardware.hasAttribute("android:name"):
                    permission_component_envinfo[hardware.getAttribute("android:name")] = 1

            dom_libraries = dom_elements.getElementsByTagName("android:name")
            for lib in dom_libraries:
                if lib.hasAttribute('android:name'):
                    permission_component_envinfo[lib.getAttribute('android:name')] = 1

            dom_sdk_versions = dom_elements.getElementsByTagName("uses-sdk")
            for sdk in dom_sdk_versions:
                if sdk.hasAttribute('android:minSdkVersion'):
                    permission_component_envinfo[sdk.getAttribute('android:minSdkVersion')] = 1
                if sdk.hasAttribute('android:targetSdkVersion'):
                    permission_component_envinfo[sdk.getAttribute('android:targetSdkVersion')] = 1
                if sdk.hasAttribute('android:maxSdkVersion'):
                    permission_component_envinfo[sdk.getAttribute('android:maxSdkVersion')] = 1

            return permission_component_envinfo
    except Exception as e:
        raise Exception("Fail to process xml file of apk {}:{}".format(apk_path, str(e)))


def get_feature_dex(apk_path, api_list):
    """
    get feature about opcode and functions, and string from .dex files
    :param apk_path: an absolute path of an apk
    :param api_list: a list of apis
    :return:  two dicts of elements {feature:occurrence, ..., } corresponds to functionality feature and string feature
    """
    string_feature_dict = defaultdict(int)
    opcode_feature_dict = defaultdict(int)
    api_feature_dict = defaultdict(int)

    try:
        _1, _2, dx = AnalyzeAPK(apk_path)
    except Exception as e:
        raise Exception('Fail to load dex file of apk {}:{}'.format(apk_path, str(e)))

    for method in dx.get_methods():
        if method.is_external():
            continue
        method_obj = method.get_method()
        for instruction in method_obj.get_instructions():
            opcode = instruction.get_name()
            opcode_feature_dict[opcode] += 1
            if 'invoke-' in opcode:
                code_body = instruction.get_output()
                if ';->' not in code_body:
                    continue
                head_part, rear_part = code_body.split(';->')
                class_name = head_part.strip().split(' ')[-1]
                method_name = rear_part.strip().split('(')[0]
                if class_name + ';->' + method_name in api_list:
                    api_feature_dict[class_name + ';->' + method_name] += 1

            if (opcode == 'const-string') or (opcode == 'const-string/jumbo'):
                code_body = instruction.get_output()
                ss_string = code_body.split(' ')[-1].strip('\'').strip('\"').encode('utf-8')
                if not ss_string:
                    continue
                hashing_string = hashlib.sha512(ss_string).hexdigest()
                string_feature_dict[hashing_string] = 1

    return string_feature_dict, api_feature_dict, opcode_feature_dict


def get_feature_shared_lib(apk_path):
    """
    get feature from ELF files
    :param apk_path: an absolute path of an apk
    :return: a dict of elements {feature:occurrence, ..., }
    """
    feature_dict = defaultdict(int)
    try:
        with zipfile.ZipFile(apk_path, 'r') as apk_zipf:
            handled_name_list = []
            for name in apk_zipf.namelist():
                if os.path.basename(name) in handled_name_list:
                    continue
                brief_file_info = magic.from_buffer(apk_zipf.read(name))
                if 'ELF' not in brief_file_info:
                    continue
                if 'ELF 64' in brief_file_info:
                    arch_info = CS_ARCH_ARM64
                elif 'ELF 32' in brief_file_info:
                    arch_info = CS_ARCH_ARM
                else:
                    raise ValueError("Android ABIs with Arm 64-bit or Arm 32 bit are support.")

                with apk_zipf.open(name, 'r') as fr:
                    bt_stream = BytesIO(fr.read())
                    elf_fhander = ELFFile(bt_stream)
                    # arm opcodes (i.e., instructions)
                    bt_code_text = elf_fhander.get_section_by_name('.text')
                    if (bt_code_text is not None) and (bt_code_text.data() is not None):
                        md = Cs(arch_info, CS_MODE_ARM)
                        for _1, _2, op_code, _3 in md.disasm_lite(bt_code_text.data(), 0x1000):
                            feature_dict[op_code] += 1
                    # functions
                    # we consider about whether a function occurs or not, rather than its frequency,
                    # for simplifying implementation.
                    # If only running the codes on linux platform, we suggest an arm supported tool 'objdump',
                    # which can be used to count the frequency of functions conveniently. We here utilize 'pyelftools'
                    # to accommodate different development platform.
                    REL_PLT_section = elf_fhander.get_section_by_name('.rela.plt')
                    if REL_PLT_section is None:
                        continue
                    symtable = elf_fhander.get_section(REL_PLT_section['sh_link'])

                    for rel_plt in REL_PLT_section.iter_relocations():
                        func_name = symtable.get_symbol(rel_plt['r_info_sym']).name
                        feature_dict[func_name] += 1

                handled_name_list.append(os.path.basename(name))
        return feature_dict
    except Exception as e:
        raise Exception("Fail to process shared library file of apk {}:{}".format(apk_path, str(e)))


def dump_feaure_list(feature_list, save_path):
    utils.dump_json(save_path, feature_list)
    return


def load_feature(save_path):
    return utils.load_json(save_path)


def wrapper_load_features(save_path):
    try:
        return load_feature(save_path)
    except Exception as e:
        return e
