from setuptools import find_packages, setup

package_name = "stand_arm_control"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="soofiyan",
    maintainer_email="soofiyan2910@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pub_arm_sub_vive = stand_arm_control.g1_arm7_sdk_pub_vive_sub:main",
        ],
    },
)
