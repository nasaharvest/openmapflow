from unittest import TestCase

try:
    from openmapflow.inference_widgets import InferenceBBox

    SKIP_TEST = False
except ModuleNotFoundError as e:
    print(e)
    SKIP_TEST = True


class TestInferenceWidgets(TestCase):
    def setUp(self) -> None:
        self.skipTest("Dependencies not installed see warning.")

    def test_inference_bbox(self):
        ibbox = InferenceBBox(min_lat=0, min_lon=0, max_lat=1, max_lon=1)
        self.assertAlmostEqual(ibbox.area, 12308.463846396064)
        self.assertEqual(ibbox.center, (0.5, 0.5))

    def test_inference_bbox_USA(self):
        ibbox = InferenceBBox(min_lat=40, min_lon=-95, max_lat=41, max_lon=-94)
        self.assertAlmostEqual(ibbox.area, 9412.650056435356)
        self.assertEqual(ibbox.center, (40.5, -94.5))
