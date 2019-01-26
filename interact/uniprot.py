"""Methods to generate genename dictionary."""
import os
import gzip
from .generic import download_from_url

UNIPROT_ID_MAPPING_URL = (
    'ftp://ftp.uniprot.org/pub/databases/uniprot/' +
    'current_release/knowledgebase/idmapping/by_organism/' +
    'HUMAN_9606_idmapping.dat.gz'
)


def dict_from_uniprot(
    key_name='UniProtKB-ID', value_name='Gene_Name', target_path=None,
    target_filename=None
):
    """
    Generate a dictionary from UniProt.

    It creates two dictionaries both using the UniProtID as key. One mapping
    UniProtID to key_name, the other UniProtID to value_name. Then
    the dictionary key_name: value_name is created. If key_name or value_name
    are 'UniProtID' then the merging step is skipped.
    """
    local_filepath = download_from_url(
        UNIPROT_ID_MAPPING_URL, target_path, target_filename=target_filename)
    key_to_id1_dict = {}
    key_to_id2_dict = {}
    with gzip.open(local_filepath) as fp:
        for line in fp:
            key, name_type, name = line.strip().decode().split('\t')
            if name_type == key_name:
                key_to_id1_dict[key] = name
            elif name_type == value_name:
                key_to_id2_dict[key] = name
    id1_to_id2_dict = {}
    if key_name == 'UniProtID':
        return key_to_id2_dict
    elif value_name == 'UniProtID':
        return {value: key for key, value in key_to_id1_dict.items()}
    else:
        for key, id1 in key_to_id1_dict.items():
            if key in key_to_id2_dict:
                id1_to_id2_dict[id1] = key_to_id2_dict[key]
        return id1_to_id2_dict
