以下是将提供的英文贡献指南翻译成中文的内容：

.. highlight:: shell

============
贡献指南
============

非常欢迎并感激您的贡献！每一点帮助都非常宝贵，我们将始终给予您相应的认可。

您可以通过以下方式贡献：

贡献类型
----------------------

报告 Bug
~~~~~~~~~~

请在 https://github.com/lsewcx/ai_modules_hub/issues 报告 Bug。

如果您报告一个 Bug，请包括以下信息：

* 您的操作系统名称和版本。
* 可能有助于排查问题的本地设置详情。
* 重现 Bug 的详细步骤。

修复 Bug
~~~~~~~~

查看 GitHub 上的 issues，任何标记为 "bug" 和 "help wanted" 的问题都开放给任何想要实现它的人。

实现功能
~~~~~~~~~~~~~~~~~~

查看 GitHub 上的 issues，任何标记为 "enhancement" 和 "help wanted" 的问题都开放给任何想要实现它的人。

编写文档
~~~~~~~~~~~~~~~~~~~

AI-Modules-Hub 始终需要更多的文档，无论是官方的 AI-Modules-Hub 文档、docstrings，还是博客文章、网络文章等。代码的文档采用谷歌风格的注释。

提交反馈
~~~~~~~~~~~~~~~

发送反馈的最佳方式是在 https://github.com/lsewcx/ai_modules_hub/issues 提交 issue。

如果您提议一个功能：

* 详细解释它的工作原理。
* 尽可能保持范围狭窄，以便于实现。
* 请记住这是一个志愿者驱动的项目，欢迎贡献。

开始吧！
------------

准备贡献？以下是如何为本地开发设置 `ai_modules_hub` 的步骤。

1. 在 GitHub 上 Fork `ai_modules_hub` 仓库。
2. 克隆您的 Fork 到本地::

    $ git clone git@github.com:your_name_here/ai_modules_hub.git

3. 安装您的本地副本到虚拟环境中。假设您已安装 virtualenvwrapper，以下是如何为您的 Fork 设置本地开发环境::

    $ mkvirtualenv ai_modules_hub
    $ cd ai_modules_hub/
    $ python setup.py develop

4. 为本地开发创建分支::

    $ git checkout -b name-of-your-bugfix-or-feature

   现在您可以在本地进行更改。

5. 完成更改后，检查您的更改是否通过了 flake8 和测试，包括使用 tox 测试其他 Python 版本::

    $ make lint
    $ make test
    或者
    $ make test-all

   要获取 flake8 和 tox，只需将它们 pip 安装到您的虚拟环境中。

6. 提交您的更改并推送您的分支到 GitHub::

    $ git add .
    $ git commit -m "您对更改的详细描述。"
    $ git push origin name-of-your-bugfix-or-feature

7. 通过 GitHub 网站提交 pull request。

Pull Request 指南
-------------------

在您提交 pull request 之前，请检查它是否符合以下指南：

1. pull request 应包含测试。
2. 如果 pull request 添加了功能，文档应该更新。将您的新功能放入一个带有 docstring 的函数中，并在 README.rst 的列表中添加该功能。
3. pull request 应该适用于 Python 3.5、3.6、3.7 和 3.8，以及 PyPy。检查 https://travis-ci.com/lsewcx/ai_modules_hub/pull_requests 并确保所有支持的 Python 版本都通过了测试。

提示
----

要运行部分测试::

    $ python -m unittest tests.test_ai_modules_hub

部署
---------

给维护者的提醒，如何部署。
确保所有更改都已提交（包括 HISTORY.rst 中的条目）。
然后运行::

    $ bump2version patch # 可能的选项：major / minor / patch
    $ git push
    $ git push --tags

Travis 将在测试通过后部署到 PyPI。

行为准则
---------------

请注意，此项目发布时附有 `贡献者行为准则`_。
通过参与此项目，您同意遵守其条款。

.. _`贡献者行为准则`: CODE_OF_CONDUCT.rst
