using System;
using System.ComponentModel;
using CodeStack.SwEx.Common.Attributes;
using CodeStack.SwEx.PMPage.Attributes;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

// Choose Lock UI

namespace TestAddin
{
    [Title("Insert Lock")]

    internal class LockPage
    {

        public enum Options_e
        {
            [Title("Left Lock")]
            Option1,
            [Title("Right Lock")]
            Option2,
        }

        [OptionBox]
        public Options_e Option { get; set; }

    }
}
