"""Fetch required files

"""

from dataclasses import dataclass
from pathlib import Path
import requests
import hashlib
import paths
import functools
from util import get_only

MAX_FETCH_ATTEMPTS = 3

@dataclass
class FileDef:
    path: Path
    remote_path: str
    hash: bytes

URL = "https://02456.gtsr.dk/{path}"
def fetch(remote_path: str, dest: Path):
    resp = requests.get(URL.format(path=remote_path))
    resp.raise_for_status()
    with dest.open("wb") as fp:
        fp.write(resp.content)
    return resp.content

def verify(expected_hash: bytes, content: bytes = None, file: Path = None) -> bool:
    if content is None and file is None:
        raise UserWarning("Either content or file must be given")
    
    if file:
        with file.open("rb") as fp:
            content = fp.read()
    
    md = hashlib.sha256()
    md.update(content)
    return expected_hash == md.digest()

def fetch_and_verify(fdef: FileDef):
    content = fetch(fdef.remote_path, fdef.path)
    attempts = 1
    while not verify(fdef.hash, content):
        if attempts >= MAX_FETCH_ATTEMPTS:
            raise UserWarning(f"Could not fetch file {fdef.path.relative_to(paths.ROOT)}. Google Drive ID: {fdef.gdrive_id}.")
        fetch(fdef.remote_path, fdef.path)
        attempts += 1


def get_by_fdef(fdef: FileDef) -> Path:
    if fdef.path.exists():
        return fdef.path
    fetch_and_verify(fdef)
    return fdef.path


def file_getter(fdef: FileDef):
    return functools.partial(get_by_fdef, fdef)
    

class Splits:
    _splits = {
        'train': '077dc2e5279860f0c705914bf7a6ea606615c36727e1866fc7872f4c59982c2d',
        'val': '8bc2e7dd921cd3f3e5ab4f609622522327c376fa104fbabc3effc99d67e135a5',
        'test': 'ee8c6e556e2844f13f59f704f67565c48f836b233ebb7fd0de21144ad30b65a1',
    }
    
    def __getitem__(self, name):
        hash = bytes.fromhex(self._splits[name])
        fdef = FileDef(paths.SPLITS_DIR / f'{name}.parquet', f'data_splits/{name}.parquet', hash)
        return get_by_fdef(fdef)
    
    def __iter__(self):
        return iter(self._splits)
    
    def __len__(self):
        return len(self._splits)

results_filtered = {
    "linear_model":                     "9a8438e4e2bbe48191c9ae9293c8c679f32c645f488e363c90f5f32cca65fe9e",
    "mini_transformer":                 "0f6bcdf35c251575b198b5c6c3bbbde614443b6baabe0e926392db4c7abc30e0",
    "small_transformer":                "d53de82b2d94c3bbbcb789b1f724ef23206b7e4fb8dd796fdfd3827d989a6020",
    "medium_transformer":               "e98e7d9cd4670b847f5c220e0a287e12fc5f43125cd4f126c110ecc5d5de9708",
    "deeper_transformer_2":             "2de0edcf501266ae5c28d508aa5cc0de507091a698597b1fe508508c4ad18b1f",
    "deeper_transformer":               "270bd5f04d85bc44ddb94b7788973c2e5fdd4bcf855801d4a82c50931175c172",
    "mini_lstm":                        "46f41d9978b6200fb84ef93dbe6070259bbb84a10d0f900a6cd58b91f467a977",
    "small_lstm":                       "bd2f97658084e8c22956772046de744e1a7c3c3c99b38e971fbc77e230868da1",
    "medium_lstm":                      "a79b0c897502aa494fdb37f873d8ba9ba8d8663caa5833e25c89d279723dd128",
    "deeper_lstm_2":                    "e3b06173e3717cea76c4fdfba0928481b5a6be409022e96be19201fbf3a55e69",
    "deeper_lstm":                      "b30dfc4ff46eac2b09b6fbbed1ffa59adf2702190108c780870403f458f91958",
    "mini_autoreg_lstm":                "593a81886c5565b0a3f7c39122db3a5195e7bfe12ad185bdfb974894d4eb3d8f",
    "small_autoreg_lstm":               "9480335c95b091278f5b86e22a9dbee1313a31f71e2e9fa33ab787bb5fde7fb0",
    "medium_autoreg_lstm":              "4ab59b18d77092c0dae04f1c4ef9103ce7e1f5dba684a35ea23050a024ee0d8a",
    "deeper_autoreg_lstm_2":            "27bf878216a1e7aa7f43d075e63825845ff52d86024a2786872257196efa995a",
    "deeper_autoreg_lstm":              "5169cc13b5f9254ccc433f6a8efd00509cdc17a9f6be96b3228ea92d5859ee8f",
    "deeper_transformer_3":             "9461ff75a9ba77db5da51a1b5fada21338c567b2a2ef416a9119c824e6ddbda7",
    "even_deeper_lstm":                 "df47013c10a62c3f017849b004b7084ad7bf1cc779eef1d389a8846ac3b71ff4",
    "ED_small_seq2seq_trans":           "35fb3eea3761d56961a9c09b02f0d52618d0123d50b91a2597ef4ba496e698dc",
    "mini_seq2seq_trans":               "01058ae322b6819ce27cccb504789962225ed68d049bfa89b9bbe82952ded8af",
    "small_seq2seq_trans":              "44d49277a704a9415aa90a50c44a1e4accf98772bc01997a8c9a165dea30f7a0",
    "medium_seq2seq_trans":             "33bd613dc8b2f9321a6b8c911949793a0b9659edd680af8e8d75e021754c726b",
    "deep_seq2seq_trans":               "c2af20d51bb5fd982affa0f2dcaaa3958905c71c7e88b3b55b4900d9a73db0bd",
    "dropoutincrease_seq2seq_trans":    "592281e787ee7d0b4e73e367adc39b3bd70e806ccecae34570cf9da64d7d290c",
    "twolayer_seq2seq_trans":           "6be8036ffed2273f94a676429947ac18a5ed897f2f76727557dd7d8fce6e76eb",
}

results_unfiltered = {
    "deeper_autoreg_lstm_2":    "2820121a76c2d860f0f3b818b663e5636749b33d0b9af34b5e51cbd6e3f0dedb",
    "deeper_autoreg_lstm":      "014fd8d2662ad1c1973c92330a2eaa3a77f02c4fae60161f0992d4f6d4ca78f1",
    "deeper_lstm_2":            "a16b135bcb2260d5f79515cf228ba814f7dbec1eb8331acaa3c06f5b86e4c0ec",
    "deeper_lstm":              "5b229b2d3e4c03a09efc9a33de25e0b105092f27aa0cccd2712e144fefe42a17",
    "deeper_transformer_2":     "aaa5939fa140d688f148f6fd08319a04cdc88771ce4a964cc7156b2168eb0c04",
    "deeper_transformer":       "a6e7b286dc434d8510723532b5a754631d0f3826ea443bc7445d9559b56bc0d2",
    "linear_model":	            "686c9949201cc7f5915c9c40c55c898125c1e36b8a30d2450ac12d1ce0e27413",
    "medium_autoreg_lstm":      "9d0a412ceb7dc7c76152542818003b18f9cb20c6567e1478e7acc41ca3aa3b50",
    "medium_lstm":              "79d593c3793f4ebee45400d0f8d67cdd2ee76c801b95f3720911ef8285673846",
    "medium_transformer":       "5e11e874f3d8c8b505e5903d86225fee4a3843cec97e80f99a3836625e6d23de",
    "mini_autoreg_lstm":        "a5bbba42d6054f8d60f8356a048811d3ec97ade01be98e656519d84517257b5e",
    "mini_lstm":                "b9b84cc807b52e79feae6d30bc24eb039250a2b8200e747d0b245f5ad615c41f",
    "mini_transformer":	        "227fbe947963c92384ee260f491f3d04aae4ea3ed0c72a021a6d8ec88f832517",
    "small_autoreg_lstm":       "5ee1ca1efa22e9db034ac495b672714a1dc708db20ac1ea07d8e147df90ce9e9",
    "small_lstm":               "d96c521bc9b301278c0105829d71dfde1ce81519cc7f01cf7bfd847a18a1b0c6",
    "small_transformer":        "06e72d46d8c3e88b550d26d2d5efdfff65aedba54c1632229912d57fe92dfdec",
}

class Results:
    def __init__(self, filtered: bool = True):
        self.filtered = filtered
        if filtered:
            self.dir = paths.RESULTS_FILTERED_DIR
            self.models = results_filtered
        else:
            self.dir = paths.RESULTS_UNFILTERED_DIR
            self.models = results_unfiltered

    def __getitem__(self, name: str):
        if name not in self.models:
            raise KeyError(f"no such model: {name}")
        remote_dir = 'results' if self.filtered else 'results_unfiltered'
        fdef = FileDef(self.dir / f'{name}_results.json', f'{remote_dir}/{name}_results.json', bytes.fromhex(self.models[name]))
        return get_by_fdef(fdef)
    
    def __iter__(self):
        return iter(self.models)
    
    def __len__(self):
        return len(self.models)


models_filtered = {
    "deeper_autoreg_lstm_2": "596f07d4f9a0f4ed9d78e478ff5f54a6093d39d900c92c39abae0ce3ea0edeed",
    "deeper_autoreg_lstm": "d72470a18506903b74c88cfc630f6b9fceefd8ae90b18cc56291d406f8828619",
    "deeper_lstm_2": "6755dc5f32b4c5ed5271f3dde4d2a38da1609858a2e9e9a2cd849dc04d1abd17",
    "deeper_lstm": "1b85a4c56d0fb6a1b7036b2e9373c0856a77d5bbd0b4b150e61376cff729f571",
    "deeper_transformer_2": "619c29cd6859c975adb698c7ccf1ec6725ffa576cd3852bdffe0fcb5984435b9",
    "deeper_transformer_3": "11c555c7e582b693f0a88b021eb4a6c52e38baa0a929204f7f1ff5f24e580c53",
    "deeper_transformer": "ed061ec36b5da9062190568f4139a052b47d0b4613e6fc47008a0e8b94279175",
    "deep_seq2seq_trans": "9ebdc13ed589c3774c61513d5bb228d7cd6bfa5fd50fe5ac0a4fae28526366ea",
    "dropoutincrease_seq2seq_trans": "d3985a698d3a7f596fde9b657fd1852a6a8f2c662fd2d22ea6c980075efaf4b7",
    "ED_small_seq2seq_trans": "7645474977750a086dd5fca7ead2a5531089b3287f842e47271b1f595c8e0aa1",
    "even_deeper_lstm": "a97e70b51b58974dbe9b71dba47a15d7b7c7eda7ad58313486a28b68e2fb1f45",
    "linear_model": "3443d7bb965df040bda2f21dce79e1cd02ba5927851310090d3aa1f715cdb9d2",
    "medium_autoreg_lstm": "9e6c19acf1fefdfcbcec195c7426299f80cea1db69fa8b9fda2d1965dc8cba80",
    "medium_lstm": "c19a3da8d965215268783d58c84a6475ad8ddae82c9eee7701f8f032cb275dc9",
    "medium_seq2seq_trans": "d5061a93a733e7dab04eec46c06c1f24f8eb82a243e6ff16ca08d94af5b9b172",
    "medium_transformer": "13331d41391ee9c2fb7dc4dd7cbdbb7bfb069279068783e43a8b54e831c21ec1",
    "mini_autoreg_lstm": "d84a1f72ec9afdfe54b7bec49c92f9552a5ae54968f5d0bac3a74042999804ff",
    "mini_lstm": "ff01de90f484e2166da02cc494c5c8f515040c5e85fb12113206e304cbfa6e0c",
    "mini_seq2seq_trans": "46a2fba3bae5e0bcfa39f19cdb50a9f21a911b2867cd5651d93dea6394cb576d",
    "mini_transformer": "47e1a3f457190c21c87ec8a29fccf0c5f5c1cf7bfcb71dbd1758824e5a12d4ab",
    "small_autoreg_lstm": "11d6c82afed26c23e509035229cb1b2acf8ac5b03e1c5a43087a1b74d3bd4798",
    "small_lstm": "82db62891955de453d968626b4e6ddbd26c0f49fefe0b04d0ebe76a9a2db13f0",
    "small_seq2seq_trans": "561daedea5b169d3704093385561851bb7d3661874e64b950d692fce52d34d51",
    "small_transformer": "8663efc6dca770984899d0d20cae58ccb44758b6a834297d562981d2665dd761",
    "twolayer_seq2seq_trans": "48a2d644cc096895db1a27a2764c506aecb15464729a496865c4852792803027",
}

models_unfiltered = {
    "deeper_autoreg_lstm_2": "4eebf63db65e98c78056f5afb318b8c1528dd7756fd184bbb356d20d2127ed93",
    "deeper_autoreg_lstm": "6e70077c3e78dba4d341cdca09ca0eb6315bbd6dd42e42ff4ad9e17d93dbaa2f",
    "deeper_lstm_2": "f2fcd324144bd2a9d1dbb15f18c9dceb65eb2d2db012e1267f47c97238956409",
    "deeper_lstm": "49166af727700fa6a6628900c6f325a5d83b0183acf2d510b36d77cbb98d3163",
    "deeper_transformer_2": "4c1f810782726276e1b271ec4ebc8e86fd638c1a8d1743e0794cb3d369ad6406",
    "deeper_transformer": "387760334c7995a32bf7b0678879abaf4c3420aec941e4908671dc301dbd1877",
    "linear_model": "8b0e0c871e6633148d93d8f20ac6243e6eb0987cc98963a50ebe7d85965ca46e",
    "medium_autoreg_lstm": "b06ca5a9fea746945e814756effaa954dd1a2f503d089db0b180e2706ca7cbf1",
    "medium_lstm": "79971fb67a605ede2718ec791031b7b268cc0b78a7309e83c73e5847bc5637ba",
    "medium_transformer": "aae27d4a073d978a3e497eb9fe582ba601341b92ecb38b85fc5f58b9c61065b4",
    "mini_autoreg_lstm": "3e124162da8128684ce0b70fbf92c8f5059b0948e49dbddb7e66a3ad92a4cf00",
    "mini_lstm": "9816d6e7bb0ba764c89e251ff85571015ae486dcd5a3c030b64b88796ba5dd24",
    "mini_transformer": "219bcf0ea61ba397862970b3013dbdc84eca5f9e4872ca857b80f2b7bdc14e9a",
    "small_autoreg_lstm": "0c382f8cc570dfe2ae23bf76f4d536cbacac7a18c083d6d916015cb4f8a2f176",
    "small_lstm": "0d9af29df20959a3c20127f66e13f13b315d7ea0a12d5beb8734fba2ae55b2d9",
    "small_transformer": "a767771bbabe1ebdccb0a6b8fff58c1308fcfa3bd33fdffba9b56fcce6c3a156",
}



class Models:
    def __init__(self, filtered: bool = True):
        self.filtered = filtered
        if filtered:
            self.dir = paths.RESULTS_FILTERED_DIR
            self.models = models_filtered
        else:
            self.dir = paths.RESULTS_UNFILTERED_DIR
            self.models = models_unfiltered

    def __getitem__(self, name: str):
        if name not in self.models:
            raise KeyError(f"no such model: {name}")
        remote_dir = 'results' if self.filtered else 'results_unfiltered'
        fdef = FileDef(self.dir / f'{name}_best.pt', f'{remote_dir}/{name}_best.pt', bytes.fromhex(self.models[name]))
        return get_by_fdef(fdef)
    
    def __iter__(self):
        return iter(self.models)
    
    def __len__(self):
        return len(self.models)
