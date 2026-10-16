import unittest
from unittest.mock import patch
from piano_refiner.remote_zip import HTTPRangeReader


class RangeReaderTests(unittest.TestCase):
    def test_chunk_order_seek_and_budget(self):
        reader=HTTPRangeReader.__new__(HTTPRangeReader)
        reader.pos=0;reader.used=0;reader.size=3*1024**2+19;reader.budget=reader.size
        source=bytes(range(251))*(reader.size//251+1)
        with patch.object(reader,'_range',side_effect=lambda start,n:source[start:start+n]) as fetch:
            self.assertEqual(reader.read(),source[:reader.size])
            self.assertEqual(fetch.call_count,4)
            self.assertEqual(reader.tell(),reader.size)
            self.assertEqual(reader.read(1),b'')
            reader.seek(0)
            with self.assertRaises(IOError):reader.read(1)


if __name__=='__main__':unittest.main()
