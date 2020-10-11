#!/usr/bin/env python3
import fire

import maskDetector
import maskDetectorHeadless


class MaskDetector:
    """Entrypoint to run edge detection tasks."""

    def maskDetector(self, noninteractive=False):
        """Run the mask detection program."""
        if noninteractive:
            maskDetectorHeadless.run()
        else:
            maskDetector.run()


if __name__ == "__main__":
    fire.Fire(MaskDetector)
