#!/usr/bin/env python3
import logging

import fire

import maskDetector
import maskDetectorHeadless


class MaskDetector:
    """Entrypoint to run edge detection tasks."""

    def maskDetector(
        self, noninteractive=False, mqtt=True, display=False, log_level="info"
    ):
        """Run the mask detection program."""

        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError("Invalid log level: %s" % log_level)
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(message)s", level=numeric_level
        )

        if noninteractive:
            logging.info("Starting Mask detector in headless mode")
            maskDetectorHeadless.run(mqtt=mqtt, display=display)
        else:
            logging.info("Starting Mask detector with graphical interface")
            maskDetector.run()


if __name__ == "__main__":
    fire.Fire(MaskDetector)
