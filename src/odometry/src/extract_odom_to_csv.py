#!/usr/bin/env python3
"""Subscribe to /path and save each pose's XYZ coordinates to CSV."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import rospy
from nav_msgs.msg import Path as PathMsg

# Change this value to control the CSV file name without ROS parameters.
CSV_FILENAME = "rooftop_lio.csv"


class PathCsvRecorder:
    """ROS node that logs nav_msgs/Path pose positions into a CSV file."""

    def __init__(self) -> None:
        self.topic = rospy.get_param('~topic', '/path')
        default_dir = Path(__file__).resolve().parent.parent / 'outputs'
        output_dir = Path(rospy.get_param('~output_dir', str(default_dir)))
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_name_param = rospy.get_param('~csv_filename', CSV_FILENAME)
        if csv_name_param:
            output_name = csv_name_param
        else:
            output_name = f"path_{timestamp}.csv"
        self.output_path = output_dir / output_name

        # Keep the file open for the node lifetime to minimize file I/O overhead.
        self._file = self.output_path.open('w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        self._writer.writerow(['x', 'y', 'z'])
        self._file.flush()

        rospy.loginfo("Writing path poses to %s", self.output_path)
        self._sub = rospy.Subscriber(self.topic, PathMsg, self._callback, queue_size=1)
        rospy.on_shutdown(self._cleanup)
        self._written_count = 0

    def _callback(self, msg: PathMsg) -> None:
        if not msg.poses:
            rospy.logdebug("Received empty Path message; skipping.")
            return

        total_poses = len(msg.poses)
        if total_poses <= self._written_count:
            # Path message did not extend beyond what we already saved, likely a repeat.
            rospy.logdebug("No new poses to record (received %d, already had %d).",
                           total_poses, self._written_count)
            return

        new_poses = msg.poses[self._written_count:]
        rows = []
        for pose_stamped in new_poses:
            pos = pose_stamped.pose.position
            rows.append((pos.x, pos.y, pos.z))
        self._writer.writerows(rows)
        self._file.flush()
        self._written_count = total_poses
        rospy.loginfo_once("Started receiving /path messages.")

    def _cleanup(self) -> None:
        rospy.loginfo("Closing CSV file %s", self.output_path)
        if hasattr(self, '_file') and not self._file.closed:
            self._file.close()


def main() -> None:
    rospy.init_node('extract_odom_to_csv')
    recorder: Optional[PathCsvRecorder] = None
    try:
        recorder = PathCsvRecorder()
        rospy.loginfo("Listening on %s. Press Ctrl-C to stop.", recorder.topic)
        rospy.spin()
    finally:
        if recorder is not None:
            recorder._cleanup()


if __name__ == '__main__':
    main()
