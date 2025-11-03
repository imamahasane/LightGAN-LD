from src.lightgan_ld.data.dummy import DummyPairs

def test_dummy_iter():
    ds = DummyPairs(n=8, img_size=64)
    s, x, y = ds[0]
    assert s.shape[0]==1 and x.shape[-1]==64 and y.shape[-1]==64
